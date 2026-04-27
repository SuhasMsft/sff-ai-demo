import os
import sys
import glob
import time
import socket
import numpy as np
import scipy as sp
import torch
import cv2
import sounddevice as sd
import soundfile as sf
import queue
from contextlib import contextmanager


def runStartupPreflight():
    """Print startup diagnostics for audio device visibility and model download connectivity."""

    print("=== Startup preflight ===")

    # Audio device visibility from container namespace
    sndPath = "/dev/snd"
    if os.path.isdir(sndPath):
        sndNodes = sorted(glob.glob(f"{sndPath}/*"))
        print(f"ALSA path found: {sndPath} ({len(sndNodes)} nodes)")
        if sndNodes:
            print("ALSA nodes:", ", ".join(os.path.basename(node) for node in sndNodes))
    else:
        print("WARNING: /dev/snd not found. Mount audio devices with: -v /dev/snd:/dev/snd")

    # Enumerate input devices through PortAudio/sounddevice
    try:
        devices = sd.query_devices()
        inputDevices = [d for d in devices if d.get("max_input_channels", 0) > 0]
        print(f"PortAudio inputs detected: {len(inputDevices)}")
        for idx, device in enumerate(inputDevices[:5]):
            print(
                f"  Input {idx + 1}: {device.get('name')} "
                f"(channels={device.get('max_input_channels')}, sample_rate={device.get('default_samplerate')})"
            )
    except Exception as err:
        print(f"WARNING: PortAudio query failed: {err}")

    # Validate model download prerequisites
    try:
        socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
        print("DNS check: huggingface.co resolved")
    except Exception as err:
        print(f"WARNING: DNS check failed for huggingface.co: {err}")
        print("Hint: try --network host if Docker bridge DNS cannot resolve external names")

    try:
        import urllib.request
        with urllib.request.urlopen("https://huggingface.co", timeout=10) as resp:
            print(f"HTTPS check: huggingface.co reachable (status={resp.status})")
    except Exception as err:
        print(f"WARNING: HTTPS check failed for huggingface.co: {err}")

    # Camera device visibility
    videoDevices = glob.glob('/dev/video*')
    if videoDevices:
        print(f"V4L2 video devices found: {len(videoDevices)} ({', '.join(sorted(videoDevices))})")
    else:
        print("WARNING: No /dev/video* devices found. Mount camera with: --device /dev/video0:/dev/video0")

    print("=== Preflight complete ===")

@contextmanager
def printVerbosely(codeBlockDescription):
    """Print to console before/after block, report outcome and duration"""

    # Print before message
    print(f"{codeBlockDescription}...")
    start = time.time()
    success = True

    try:
        yield
    except Exception:
        success = False
        raise

    # Print after message in green or red

    finally:
        elapsed = time.time() - start
        COLORS = {
            "G" : "\033[92m", # Green
            "R" : "\033[91m", # Red
            "W" : "\033[0m", # Default, usually white
        }
        if success:
            print(f"{COLORS['G']}✔ {codeBlockDescription} succeeded in {elapsed:.2f}s{COLORS['W']}")
        else:
            print(f"{COLORS['R']}✗ {codeBlockDescription} failed after {elapsed:.2f}s{COLORS['W']}")

@contextmanager
def suppressStdOut():
    """Suppress stdout and stderr at the OS file descriptor level (catches C++ output from OpenCV, NeMo, etc)"""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(stdout_fd)
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stdout_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stdout, stdout_fd)
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stdout)
            os.close(old_stderr)

def listMicrophones():
    """Return available microphones as dictionaries of index, name, channels, default_samplerate"""
    devices = sd.query_devices()

    if not devices:
        return None

    allMicrophones = []
    for index, device in enumerate(devices):

        # Skip devices with no input channels
        if device["max_input_channels"] <= 0:
            continue

        # Skip virtual/system devices with suspiciously high channel counts
        if device["max_input_channels"] > 8:
            continue

        # Skip devices with no default sample rate
        if not device.get("default_samplerate"):
            continue

        allMicrophones.append({
            "Name": device["name"],
            "Index": index,
            "Channels": device["max_input_channels"],
            "SampleRate": device["default_samplerate"]
        })

    return allMicrophones or None

def selectBestMicrophone(allMicrophones):
    """
    Given list of microphones, select
        1. The first that includes "USB" in the name, else
        2. The first that has exactly 1 input audio channel, else
        3. The first overall.
    """

    if not allMicrophones:
        return None

    # 1. Prefer single-channel microphones
    monoMics = [mic for mic in allMicrophones if mic["Channels"] == 1]
    if monoMics:
        return monoMics[0]

    # 2 Prefer USB microphones
    usbMics = [mic for mic in allMicrophones if "usb" in mic["Name"].lower()]
    if usbMics:
        return usbMics[0]

    # 3. Fallback: return the first device
    return allMicrophones[0]

def listCameras():
    """Return available cameras as dictionaries of index, width, height via V4L2"""
    allCameras = []
    videoDevices = sorted(glob.glob('/dev/video*'))
    for device in videoDevices:
        index = int(device.replace('/dev/video', ''))
        with suppressStdOut():
            camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if camera.isOpened():
                success, frame = camera.read()
                if success and frame is not None:
                    allCameras.append({
                        "Index": index,
                        "Width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "Height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    })
                camera.release()
    return allCameras or None

def selectBestCamera(allCameras):
    """Select highest resolution camera, or first available."""
    if not allCameras:
        return None
    return sorted(allCameras, key=lambda c: c["Width"] * c["Height"], reverse=True)[0]

def initialize():
    global GPU, CAMERA, MICROPHONE_INDEX, MICROPHONE_SAMPLERATE, SPEECH_MODEL
    global VISION_MODEL, VISION_PROCESSOR, LLM_TOKENIZER, LLM_MODEL

    # GPU
    with printVerbosely("Check CUDA availability"):
        if not torch.cuda.is_available():
            print("FATAL: GPU required but not found.")
            print("Check: nvidia-smi, /dev/nvidia0, udev rule 99-nvidia-device-nodes.rules")
            sys.exit(1)
        GPU = torch.device("cuda:0")
        print(f"Using device: {GPU}")

    # CAMERA
    with printVerbosely("Prepare camera"):
        allCameras = listCameras()
        if allCameras:
            bestCamera = selectBestCamera(allCameras)
            cameraIndex = int(bestCamera["Index"])
            CAMERA = cv2.VideoCapture(cameraIndex, cv2.CAP_V4L2)
            if CAMERA.isOpened():
                CAMERA.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"Selected camera: /dev/video{cameraIndex} ({bestCamera['Width']}x{bestCamera['Height']})")
            else:
                print("WARNING: Camera found but could not open. Vision disabled.")
                CAMERA = None
        else:
            print("WARNING: No camera found. Vision disabled — speech-only mode.")
            CAMERA = None

    # MICROPHONE
    with printVerbosely("Prepare microphone"):
        allMicrophones = listMicrophones()
        if not allMicrophones:
            raise RuntimeError("No microphone found")
        bestMicrophone = selectBestMicrophone(allMicrophones)
        print(f"Selected microphone: {bestMicrophone['Name']} (Index: {bestMicrophone['Index']}, Channels: {bestMicrophone['Channels']}, SampleRate: {bestMicrophone['SampleRate']})")
        MICROPHONE_INDEX = int(bestMicrophone["Index"])
        MICROPHONE_SAMPLERATE = int(bestMicrophone["SampleRate"])

    # VISION MODEL (Grounding-DINO)
    if CAMERA is not None:
        with printVerbosely("Load vision model: grounding-dino-base"):
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            modelId = "IDEA-Research/grounding-dino-base"
            VISION_PROCESSOR = AutoProcessor.from_pretrained(modelId)
            VISION_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(modelId).to(GPU)
            VISION_MODEL.eval()
    else:
        VISION_MODEL = None
        VISION_PROCESSOR = None

    # LANGUAGE MODEL (Qwen — runs on CPU, small enough)
    with printVerbosely("Load language model: qwen2.5-0.5b-instruct"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        llmId = "Qwen/Qwen2.5-0.5B-Instruct"
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(llmId)
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(llmId).to("cpu")
        LLM_MODEL.eval()

    # SPEECH MODEL (Parakeet — GPU if available)
    with printVerbosely("Load speech model: parakeet-tdt-0.6b-v2"):
        import nemo.collections.asr as nemo_asr
        SPEECH_MODEL = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        SPEECH_MODEL = SPEECH_MODEL.to(GPU)
        print(f"Speech model on: {GPU}")

def makeSpeechCallback(utteranceQueue, minTalkingThreshold=10, maxSilenceThreshold=10, quietThreshold=0.01):
    from collections import deque
    MAX_BUFFER_CHUNKS = 300  # ~30 seconds at 48kHz/10 blocksize
    audioBuffer = deque(maxlen=MAX_BUFFER_CHUNKS)
    silenceCount = 0
    talkingCount = 0

    def callback(indata, frames, callback_time, status):
        """Callback that buffers speech, segments utterances, and pushes segmented audio to a queue."""
        nonlocal audioBuffer, silenceCount, talkingCount

        def isTooQuiet(chunk, threshold=0.03):
            rms = np.sqrt(np.mean(chunk ** 2))
            return rms < threshold

        # Extract, downsample, add to buffer
        audioChunk = indata[:, 0].astype(np.float32)
        audioChunkDownsampled = sp.signal.resample_poly(audioChunk, up=1, down=3)
        audioBuffer.append(audioChunkDownsampled)

        # Use energy threshold to track speaking vs silence in callback thread.
        if isTooQuiet(audioChunkDownsampled, threshold=quietThreshold):
            silenceCount += 1
        else:
            talkingCount += 1
            silenceCount = 0

        # If there hasn't been maxSilenceThreshold of consecutive silence yet, keep listening
        if silenceCount < maxSilenceThreshold:
            return

        # If we've hit maxSilenceThreshold, was there also enough talking? If yes, queue full utterance audio
        if talkingCount >= minTalkingThreshold and len(audioBuffer) >= (minTalkingThreshold + maxSilenceThreshold):
            fullAudio = np.concatenate(audioBuffer, axis=0)
            utteranceQueue.put(fullAudio)

        # Whether or not there was talking, start over
        audioBuffer.clear()
        silenceCount = 0
        talkingCount = 0

    return callback

def transcribeSpeechCommand(speechModel, audio):
    """Transcribe one utterance audio buffer into text."""
    return speechModel.transcribe([audio], verbose=False)[0].text.strip()

def extractObjectFromTranscription(fullText):
    """Use Qwen to extract the searchable object phrase from transcribed speech."""
    prompt = f"""You extract only the searchable object phrase from a spoken request.

Rules:
1) If a color is present, you MUST keep the color in the output.
2) Never drop color words. "blue cube" must stay "blue cube", not "cube".
3) Remove filler words like "show me", "can you", "please", "I want", "look for".
4) Return a short noun phrase only, lowercase, with no punctuation.
5) If no color is provided, return only the object words.

Examples:
- "show me the blue cube" -> "blue cube"
- "can you find a red ball please" -> "red ball"
- "look for the yellow toy car" -> "yellow toy car"
- "find the cube" -> "cube"

Sentence: {fullText}
Output:"""

    with torch.inference_mode():
        inputs = LLM_TOKENIZER(prompt, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        outputs = LLM_MODEL.generate(**inputs, max_new_tokens=20, temperature=0.3)
        response = LLM_TOKENIZER.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    response = response.split('\n')[0].strip()
    return response if response else fullText

def lookForObject(query, threshold=0.1):
    """Capture a camera frame, detect objects matching query, return (x,y) center or None."""
    if CAMERA is None or VISION_MODEL is None:
        return None

    knownColors = {"red": 0, "yellow": 30, "green": 60, "blue": 120}

    def extractColorFromQuery(q):
        for color in knownColors:
            if color in q.lower():
                return color
        return None

    def computeColorDistance(crop, targetColor):
        cropHSL = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cropAvgHue = float(np.mean(cropHSL[:, :, 0]))
        return min(abs(cropAvgHue - targetColor), 180 - abs(cropAvgHue - targetColor))

    # Discard buffered frames
    CAMERA.grab()
    CAMERA.grab()
    CAMERA.grab()

    success, frame = CAMERA.read()
    if not success:
        print("WARNING: Failed to read camera frame")
        return None

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameH, frameW = frame.shape[:2]

    # Object detection via Grounding-DINO
    inputs = VISION_PROCESSOR(images=frameRGB, text=query, return_tensors="pt")
    inputs = {k: (v.to(GPU) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = VISION_MODEL(**inputs)

    targetSizes = torch.tensor([frameRGB.shape[:2]], device=GPU)
    results = VISION_PROCESSOR.post_process_grounded_object_detection(
        outputs, target_sizes=targetSizes, threshold=threshold
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.int().tolist()
        x1, x2 = max(0, x1), min(frameW - 1, x2)
        y1, y2 = max(0, y1), min(frameH - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        boxArea = (x2 - x1) * (y2 - y1)
        if boxArea > 0.5 * frameH * frameW:
            continue
        # Center crop for color detection
        bw, bh = x2 - x1, y2 - y1
        cx1, cx2 = x1 + bw // 4, x1 + (bw * 3) // 4
        cy1, cy2 = y1 + bh // 4, y1 + (bh * 3) // 4
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        detections.append({"score": float(score), "label": label, "box": (x1, y1, x2, y2), "crop": crop})

    if not detections:
        return None

    queryColor = extractColorFromQuery(query)
    if queryColor is None:
        bestDetection = max(detections, key=lambda d: d["score"])
    else:
        for det in detections:
            det["colorDistance"] = computeColorDistance(det["crop"], knownColors[queryColor])
        bestDetection = min(detections, key=lambda d: d["colorDistance"])

    x1, y1, x2, y2 = bestDetection["box"]
    centerX, centerY = (x1 + x2) // 2, (y1 + y2) // 2
    print(f"  Detected '{bestDetection['label']}' at ({centerX}, {centerY}) confidence={bestDetection['score']:.2f}")
    return (centerX, centerY)


def transitionToListeningState():
    global state, query, lookUntil

    state = STATE_LISTENING
    print("Listening...")
    lookUntil = 0.0
    query = None


def transitionToLookingState():
    global state, query, lookUntil

    state = STATE_LOOKING
    print(f"Looking for: '{query}'...")
    lookUntil = time.time() + 10.0

################## MAIN CODE ##################

GPU = None

# Vision
CAMERA = None
VISION_MODEL = None
VISION_PROCESSOR = None

# Language
LLM_TOKENIZER = None
LLM_MODEL = None

# Speech
MICROPHONE_INDEX = None
MICROPHONE_SAMPLERATE = None
SPEECH_MODEL = None

runStartupPreflight()
initialize()

utteranceQueue = queue.Queue()
SPEECH_CALLBACK = makeSpeechCallback(utteranceQueue)

# State
STATE_LISTENING = "LISTENING"
STATE_LOOKING = "LOOKING"

state = STATE_LISTENING
query = None
lookUntil = 0.0

transitionToListeningState()

# Open microphone stream
with sd.InputStream(device=MICROPHONE_INDEX, channels=1, samplerate=MICROPHONE_SAMPLERATE, blocksize=int(MICROPHONE_SAMPLERATE / 10), dtype="float32", callback=SPEECH_CALLBACK):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            if state == STATE_LISTENING:
                try:
                    utteranceAudio = utteranceQueue.get_nowait()
                    fullText = transcribeSpeechCommand(SPEECH_MODEL, utteranceAudio)
                    if fullText:
                        print(f"Heard: '{fullText}'")
                        extractedObject = extractObjectFromTranscription(fullText)
                        print(f"Interpreted: '{extractedObject}'")
                        query = extractedObject
                        transitionToLookingState()
                except queue.Empty:
                    pass

            elif state == STATE_LOOKING:
                if now >= lookUntil:
                    transitionToListeningState()
                elif query:
                    result = lookForObject(query)
                    if result:
                        print(f"Found at (x, y) = {result}")
                    time.sleep(0.4)  # Throttle to ~2-3 FPS to avoid GPU saturation

            sd.sleep(50)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if CAMERA is not None:
            CAMERA.release()
        print("Cleanup complete.")
