"""
llm_camera_zoom.py
Language-controlled camera: YOLOv8 detection+tracking, text commands -> zoom.

Run:
  pip install ultralytics opencv-python
  python llm_camera_zoom.py

Keys:
  Type commands in the terminal while the window is open (e.g., "zoom on the table").
  Press ESC in the video window to quit.
"""

import cv2
import math
import threading
import queue
import re
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

# -----------------------------
# 0) Config
# -----------------------------
CAM_SOURCE = 1               # 0 for default webcam; or RTSP/HTTP URL
MODEL_PATH = "yolov8n.pt"    # swap to yolov8s.pt for better accuracy
CONF_THR = 0.5               # confidence threshold for detections
DEFAULT_ZOOM = 1.8           # default crop zoom level

# Aliases to bridge user words to detector class names
# COCO uses "dining table", but people say "table", etc.
CLASS_ALIASES = {
    "person": ["person", "human"],
    "table": ["table", "desk", "workbench", "dining table", "coffee table"],
    "chair": ["chair", "seat"],
    "cup": ["cup", "mug"],
    "bottle": ["bottle", "water bottle"],
    "tv": ["tv", "monitor", "television"],
    "laptop": ["laptop", "notebook"],
    "cell phone": ["phone", "cell", "cell phone", "smartphone"],
    # add your own as needed
}

# -----------------------------
# 1) Detector & tracker setup
# -----------------------------
cap = cv2.VideoCapture(CAM_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera source: {CAM_SOURCE}")

model = YOLO(MODEL_PATH)
NAMES = model.names

def yolo_frame():
    """Grab a frame and run YOLO tracking; return (frame, detections)."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, []

    # Ultralytics tracking (ByteTrack) with persist=True to keep IDs stable
    results = model.track(source=frame, stream=False, persist=True, verbose=False, conf=CONF_THR)
    dets = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        ids   = boxes.id.cpu().tolist() if boxes.id is not None else [None]*len(boxes)
        clses = boxes.cls.cpu().tolist() if boxes.cls is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        xyxy  = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
        for i in range(len(xyxy)):
            cid = int(clses[i])
            name = NAMES.get(cid, str(cid))
            dets.append({
                "id": int(ids[i]) if ids[i] is not None else -1,
                "cls": name,
                "conf": float(confs[i]),
                "bbox": xyxy[i],  # [x1,y1,x2,y2]
            })
    return frame, dets

# -----------------------------
# 2) Command parsing (rule-based; LLM can replace later)
# -----------------------------
def resolve_class(user_word: str) -> str:
    uw = user_word.lower().strip()
    for canon, synonyms in CLASS_ALIASES.items():
        if uw in synonyms:
            return canon
    return uw

TARGET_PATTERNS = [
    r"(?:on|onto|to|the|a)\s+([a-zA-Z0-9#]+)$",  # Modified to accept numbers and #
    r"(?:zoom|focus|center)\s+(?:on|onto)?\s*([a-zA-Z0-9#]+)$",
]

def parse_command(cmd: str) -> Dict[str, Any]:
    """
    Return {"action": "zoom|focus|center|track|stop", "target": str, "which": "...", "level": float}
    """
    text = cmd.strip().lower()

    # action
    if any(w in text for w in ["stop", "cancel", "reset"]):
        action = "stop"
    elif any(w in text for w in ["center", "recenter", "center on"]):
        action = "center"
    elif any(w in text for w in ["track", "follow"]):
        action = "track"
    elif any(w in text for w in ["zoom", "focus"]):
        action = "zoom"
    else:
        action = "zoom"

    # which (instance selection)
    which = "largest"
    if "left" in text: which = "left"
    if "right" in text: which = "right"
    if "nearest" in text or "closest" in text: which = "nearest"
    m = re.search(r"id\s*=\s*(\d+)", text)
    if m:
        which = f"id={int(m.group(1))}"

    # zoom level like "2x" or "zoom 2"
    level = None
    m = re.search(r"(\d+(\.\d+)?)\s*x", text)
    if m:
        try: level = float(m.group(1))
        except: pass
    if level is None:
        m2 = re.search(r"(?:zoom|focus)\s+(\d+(\.\d+)?)", text)
        if m2:
            try: level = float(m2.group(1))
            except: pass
    if level is None:
        level = DEFAULT_ZOOM

    # target class (very simple heuristic)
    target = None
    for pat in TARGET_PATTERNS:
        mm = re.search(pat, text)
        if mm:
            candidate = mm.group(1).strip()
            # Handle object#id format
            if "#" in candidate:
                parts = candidate.split("#")
                target = resolve_class(parts[0])
                which = f"id={parts[1]}"
            else:
                # Trim trailing articles/punctuation
                candidate = re.sub(r"[^a-zA-Z ]", "", candidate).strip()
                target = resolve_class(candidate)
            break
    if target is None:
        # fallback: last word
        target = resolve_class(text.split()[-1])

    return {"action": action, "target": target, "which": which, "level": float(level)}

# -----------------------------
# 3) Box selection helpers
# -----------------------------
def area(b):
    x1, y1, x2, y2 = b
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

def center(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def pick_box(dets: List[Dict[str, Any]], target: str, which: str, fw: int, fh: int):
    # Build a target name set for loose matching
    tset = set([target])
    tset |= set(CLASS_ALIASES.get(target, []))

    # relaxed match: exact, alias, or substring containment
    cand = [
        d for d in dets
        if d["conf"] >= CONF_THR and (
            d["cls"] in tset or target in d["cls"] or any(a in d["cls"] for a in tset)
        )
    ]
    if not cand:
        return None

    if which.startswith("id="):
        try:
            tid = int(which.split("=")[1])
            for d in cand:
                if d["id"] == tid:
                    return d
        except:
            pass  # fall through to other strategies

    if which == "largest":
        return max(cand, key=lambda d: area(d["bbox"]))
    if which == "nearest":
        cx, cy = fw / 2.0, fh / 2.0
        return min(cand, key=lambda d: (center(d["bbox"])[0] - cx) ** 2 + (center(d["bbox"])[1] - cy) ** 2)
    if which == "left":
        return min(cand, key=lambda d: center(d["bbox"])[0])
    if which == "right":
        return max(cand, key=lambda d: center(d["bbox"])[0])
    return cand[0]

# -----------------------------
# 4) Digital crop-zoom (works on any webcam)
# -----------------------------
def crop_zoom(frame, bbox, zoom_level=1.8):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Expand around center to keep context
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = max(1, int((x2 - x1) * zoom_level)), max(1, int((y2 - y1) * zoom_level))
    x1n, y1n = max(0, cx - bw // 2), max(0, cy - bh // 2)
    x2n, y2n = min(w, cx + bw // 2), min(h, cy + bh // 2)
    crop = frame[y1n:y2n, x1n:x2n]
    if crop.size == 0:
        return frame
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

# -----------------------------
# 5) Interactive loop
# -----------------------------
cmd_q: "queue.Queue[str]" = queue.Queue()

def input_thread():
    while True:
        try:
            s = input("> ")
            if s is None:
                break
            cmd_q.put(s)
        except (EOFError, KeyboardInterrupt):
            break

threading.Thread(target=input_thread, daemon=True).start()

selected_id: Optional[int] = None
intent: Optional[Dict[str, Any]] = None

print("🎥 Running. Type commands like: 'zoom onto chair#2' or 'reset' to return to original view.")
print("Press ESC in the video window to quit.")

while True:
    frame, dets = yolo_frame()
    if frame is None:
        print("Camera frame not available. Exiting.")
        break

    H, W = frame.shape[:2]

    # Process any new commands
    while not cmd_q.empty():
        raw = cmd_q.get()
        parsed = parse_command(raw)
        if parsed["action"] == "stop":
            intent = None
            selected_id = None
            print("✔ Reset to original view.")
        else:
            intent = parsed
            selected_id = None
            print(f"✔ Intent: {intent}")

    target_det = None
    if intent:
        # keep lock on the same object id if still visible
        if selected_id is not None:
            for d in dets:
                if d["id"] == selected_id:
                    target_det = d
                    break
        # else (or if lost), pick a new one matching the command
        if target_det is None:
            target_det = pick_box(dets, intent["target"], intent["which"], W, H)
            selected_id = target_det["id"] if target_det else None

    # draw overlays
    view = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        is_sel = (target_det is not None and d is target_det)
        color = (0, 255, 0) if is_sel else (255, 255, 255)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        label = f'{d["cls"]}#{d["id"]} {d["conf"]:.2f}'
        cv2.putText(view, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # apply digital zoom/crop when asked
    if intent and intent["action"] in ("zoom", "focus") and target_det:
        lvl = max(1.0, float(intent.get("level", DEFAULT_ZOOM)))
        view = crop_zoom(view, target_det["bbox"], lvl)
    elif intent and intent["action"] == "center" and target_det:
        # "center" = gentle crop (minimal zoom)
        view = crop_zoom(view, target_det["bbox"], 1.2)

    # show
    cv2.imshow("LLM-Controlled Zoom (YOLO front-end)", view)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC = quit
        break

cap.release()
cv2.destroyAllWindows()
