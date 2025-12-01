import cv2
import math
import threading
import queue
import re
import os
import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

DEPTH_ANYTHING_AVAILABLE = False
depth_model = None
torch_available = False

try:
    import torch
    from PIL import Image
    import numpy as np
    torch_available = True
    
    try:
        import sys
        da3_path = os.path.join(os.path.dirname(__file__), "Depth-Anything-3", "src")
        if os.path.exists(da3_path):
            sys.path.insert(0, da3_path)
        
        from depth_anything_3.api import DepthAnything3
        DEPTH_ANYTHING_AVAILABLE = True
        print("✓ Depth Anything 3 library found")
    except ImportError as e:
        DEPTH_ANYTHING_AVAILABLE = False
        print(f"⚠️  Depth Anything 3 not available: {e}")
        print("   Using placeholder depth visualization.")
        
except ImportError:
    print("⚠️  PyTorch not available. Install torch for Depth Anything 3 support.")
    print("   Install: pip install torch torchvision pillow")

#Congifg
CAM_SOURCE = 1
MODEL_PATH = "yolov8n.pt"
CONF_THR = 0.5
DEFAULT_ZOOM = 1.8

# Modes
MODE_YOLO = 1
MODE_DEPTH = 2
MODE_EDGE = 3
MODE_ALL = 4
current_mode = MODE_YOLO

# Interaction modes
INTERACTION_PERCEPTION = "perception"  # Switch camera modes (depth, edge, yolo, etc.)
INTERACTION_YOLO_ZOOM = "yolo_zoom"   # YOLO zoom mode (terminal commands)
current_interaction_mode = INTERACTION_PERCEPTION  # Start in perception mode

CLASS_ALIASES = {
    "person": ["person", "human"],
    "table": ["table", "desk", "workbench", "dining table", "coffee table"],
    "chair": ["chair", "seat"],
    "cup": ["cup", "mug"],
    "bottle": ["bottle", "water bottle"],
    "tv": ["tv", "monitor", "television"],
    "laptop": ["laptop", "notebook"],
    "cell phone": ["phone", "cell", "cell phone", "smartphone"],
}

# Detector & tracker setup
cap = cv2.VideoCapture(CAM_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera source: {CAM_SOURCE}")

model = YOLO(MODEL_PATH)
NAMES = model.names

# Initialize Depth Anything 3 model if available
DA3_MODEL_NAME = "da3-large"
DA3_MODEL_DIR = os.environ.get("DA3_MODEL_DIR", None)

if DEPTH_ANYTHING_AVAILABLE and torch_available:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Depth Anything 3 on {device}...")
        
        if DA3_MODEL_DIR and os.path.exists(DA3_MODEL_DIR):
            # Load from local directory
            depth_model = DepthAnything3.from_pretrained(DA3_MODEL_DIR)
            print(f"✓ Loaded DA3 model from {DA3_MODEL_DIR}")
        else:
            try:
                depth_model = DepthAnything3(model_name=DA3_MODEL_NAME)
                print(f"✓ Created DA3 model: {DA3_MODEL_NAME}")
                print("⚠️  Note: Model weights need to be loaded separately")
                print("   Set DA3_MODEL_DIR environment variable or use from_pretrained()")
            except Exception as e:
                print(f"⚠️  Could not create DA3 model: {e}")
                print("   You may need to download model weights from Hugging Face")
                DEPTH_ANYTHING_AVAILABLE = False
                depth_model = None
        
        if depth_model is not None:
            depth_model = depth_model.to(device)
            depth_model.eval()
            print("✓ Depth Anything 3 model initialized and ready")
    except Exception as e:
        print(f"⚠️  Could not initialize Depth Anything 3: {e}")
        import traceback
        traceback.print_exc()
        DEPTH_ANYTHING_AVAILABLE = False
        depth_model = None

def get_frame():
    """Grab a frame from camera."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame

def yolo_frame():
    """Grab a frame and run YOLO tracking; return (frame, detections)."""
    frame = get_frame()
    if frame is None:
        return None, []

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
                "bbox": xyxy[i],
            })
    return frame, dets

def depth_frame(frame):

    h, w = frame.shape[:2]
    
    if not DEPTH_ANYTHING_AVAILABLE or depth_model is None or not torch_available:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_like = cv2.edgePreservingFilter(frame, flags=1, sigma_s=50, sigma_r=0.4)
        gray_depth = cv2.cvtColor(depth_like, cv2.COLOR_BGR2GRAY)
        
        # Normalize and invert (closer = brighter = warmer colors)
        gray_norm = cv2.normalize(gray_depth, None, 0, 255, cv2.NORM_MINMAX)
        gray_inv = 255 - gray_norm
        
        depth_colored = cv2.applyColorMap(gray_inv, cv2.COLORMAP_TURBO)
        return depth_colored
    
    try:
        # Depth Anything 3 processing using the API
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        with torch.no_grad():
            prediction = depth_model.inference(
                [pil_image],
                process_res=504,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
        
        if prediction.depth is not None and len(prediction.depth) > 0:
            depth_np = prediction.depth[0]
            
            if isinstance(depth_np, torch.Tensor):
                depth_np = depth_np.cpu().numpy()
            
            if depth_np.shape != (h, w):
                depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
            

            depth_min, depth_max = depth_np.min(), depth_np.max()
            
            if depth_max > depth_min:
                depth_normalized = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
                depth_inverted = 255.0 - depth_normalized
            else:
                depth_inverted = np.zeros_like(depth_np)
            
            depth_inverted = depth_inverted.astype(np.uint8)
            
            depth_colored = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_TURBO)
            
            return depth_colored
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_like = cv2.edgePreservingFilter(frame, flags=1, sigma_s=50, sigma_r=0.4)
            gray_depth = cv2.cvtColor(depth_like, cv2.COLOR_BGR2GRAY)
            gray_norm = cv2.normalize(gray_depth, None, 0, 255, cv2.NORM_MINMAX)
            gray_inv = 255 - gray_norm
            return cv2.applyColorMap(gray_inv, cv2.COLORMAP_TURBO)
            
    except Exception as e:
        print(f"Depth Anything 3 error: {e}")
        import traceback
        traceback.print_exc()
        # Improved fallback visualization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_like = cv2.edgePreservingFilter(frame, flags=1, sigma_s=50, sigma_r=0.4)
        gray_depth = cv2.cvtColor(depth_like, cv2.COLOR_BGR2GRAY)
        gray_norm = cv2.normalize(gray_depth, None, 0, 255, cv2.NORM_MINMAX)
        gray_inv = 255 - gray_norm
        return cv2.applyColorMap(gray_inv, cv2.COLORMAP_TURBO)

def edge_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # Convert to BGR for display
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def get_yolo_view(frame, dets, intent, selected_id, target_det):
    """Get YOLO view with detections and zoom."""
    H, W = frame.shape[:2]
    view = frame.copy()
    
    # Draw overlays
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        is_sel = (target_det is not None and d is target_det)
        color = (0, 255, 0) if is_sel else (255, 255, 255)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        label = f'{d["cls"]}#{d["id"]} {d["conf"]:.2f}'
        cv2.putText(view, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Apply digital zoom/crop when asked
    if intent and intent["action"] in ("zoom", "focus") and target_det:
        lvl = max(1.0, float(intent.get("level", DEFAULT_ZOOM)))
        view = crop_zoom(view, target_det["bbox"], lvl)
    elif intent and intent["action"] == "center" and target_det:
        view = crop_zoom(view, target_det["bbox"], 1.2)
    
    # Add label
    cv2.putText(view, "YOLO Detection", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return view


#Command parsing (rule-based; LLM can replace later)

def resolve_class(user_word: str) -> str:
    uw = user_word.lower().strip()
    for canon, synonyms in CLASS_ALIASES.items():
        if uw in synonyms:
            return canon
    return uw

TARGET_PATTERNS = [
    r"(?:on|onto|to|the|a)\s+([a-zA-Z0-9#]+)$",
    r"(?:zoom|focus|center)\s+(?:on|onto)?\s*([a-zA-Z0-9#]+)$",
]

def parse_command(cmd: str) -> Dict[str, Any]:
    """
    Return {"action": "zoom|focus|center|track|stop", "target": str, "which": "...", "level": float}
    """
    text = cmd.strip().lower()
    
    # Handle empty input
    if not text:
        return {"action": "stop", "target": None, "which": "largest", "level": DEFAULT_ZOOM}

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

    # which
    which = "largest"
    if "left" in text: which = "left"
    if "right" in text: which = "right"
    if "nearest" in text or "closest" in text: which = "nearest"
    m = re.search(r"id\s*=\s*(\d+)", text)
    if m:
        which = f"id={int(m.group(1))}"

    # zoom level
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

    target = None
    for pat in TARGET_PATTERNS:
        mm = re.search(pat, text)
        if mm:
            candidate = mm.group(1).strip()
            if "#" in candidate:
                parts = candidate.split("#")
                target = resolve_class(parts[0])
                which = f"id={parts[1]}"
            else:
                # Trim trailing articles/punctuation
                candidate = re.sub(r"[^a-zA-Z ]", "", candidate).strip()
                target = resolve_class(candidate)
            break
    
    # Try to extract target from text if not found by patterns
    if target is None:
        words = text.split()
        if words:
            # Try to find a meaningful target (skip common words)
            skip_words = {"zoom", "focus", "on", "onto", "to", "the", "a", "in", "into", "center", "track", "stop", "cancel", "reset"}
            for word in reversed(words):  # Start from the end
                if word not in skip_words:
                    target = resolve_class(word)
                    break
            # If still no target, use the last word
            if target is None:
                target = resolve_class(words[-1])
        else:
            # No words found, use a default
            target = "person"  # Default fallback

    return {"action": action, "target": target, "which": which, "level": float(level)}

#Box selection helpers

def area(b):
    x1, y1, x2, y2 = b
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

def center(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def pick_box(dets: List[Dict[str, Any]], target: str, which: str, fw: int, fh: int):
    tset = set([target])
    tset |= set(CLASS_ALIASES.get(target, []))

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
            pass

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


#Digital crop-zoom (works on any webcam)

def crop_zoom(frame, bbox, zoom_level=1.8):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = max(1, int((x2 - x1) * zoom_level)), max(1, int((y2 - y1) * zoom_level))
    x1n, y1n = max(0, cx - bw // 2), max(0, cy - bh // 2)
    x2n, y2n = min(w, cx + bw // 2), min(h, cy + bh // 2)
    crop = frame[y1n:y2n, x1n:x2n]
    if crop.size == 0:
        return frame
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

#Interactive loop
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

print("🎥 Running. Interaction Modes:")
print("  Press 'm' to enter Perception Change Mode (switch camera modes)")
print("  Press 'y' to enter YOLO Zoom Mode (terminal commands for zoom)")
print("")
print("📹 In Perception Mode: Press '1' for YOLO, '2' for DepthAnything, '3' for Edge Detection, '4' for All Views")
print("🔍 In YOLO Zoom Mode: Type commands like 'zoom onto chair#2' or 'reset' in terminal")
print("Press ESC in the video window to quit.")

while True:
    H, W = 0, 0
    view = None
    dets = []
    frame = None

    # Process any new commands (only in YOLO zoom mode and YOLO/ALL camera modes)
    if current_interaction_mode == INTERACTION_YOLO_ZOOM and (current_mode == MODE_YOLO or current_mode == MODE_ALL):
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

    # Process based on current mode
    if current_mode == MODE_YOLO:
        # YOLO mode: detection and tracking
        frame, dets = yolo_frame()
        if frame is None:
            print("Camera frame not available. Exiting.")
            break
        H, W = frame.shape[:2]
        
        target_det = None
        if intent:
            if selected_id is not None:
                for d in dets:
                    if d["id"] == selected_id:
                        target_det = d
                        break
            if target_det is None:
                target_det = pick_box(dets, intent["target"], intent["which"], W, H)
                selected_id = target_det["id"] if target_det else None
        
        view = get_yolo_view(frame, dets, intent, selected_id, target_det)
            
    elif current_mode == MODE_DEPTH:
        # DepthAnything mode
        frame = get_frame()
        if frame is None:
            print("Camera frame not available. Exiting.")
            break
        H, W = frame.shape[:2]
        view = depth_frame(frame)
        # Add mode label
        cv2.putText(view, "DepthAnything", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
    elif current_mode == MODE_EDGE:
        # Edge detection mode
        frame = get_frame()
        if frame is None:
            print("Camera frame not available. Exiting.")
            break
        H, W = frame.shape[:2]
        view = edge_frame(frame)
        # Add mode label
        cv2.putText(view, "Edge Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    elif current_mode == MODE_ALL:
        # All views mode: show YOLO, Depth, and Edge side by side
        base_frame = get_frame()
        if base_frame is None:
            print("Camera frame not available. Exiting.")
            break
        H, W = base_frame.shape[:2]
        
        # Get YOLO detections on the same frame
        results = model.track(source=base_frame, stream=False, persist=True, verbose=False, conf=CONF_THR)
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
                    "bbox": xyxy[i],
                })
        
        target_det = None
        if intent:
            if selected_id is not None:
                for d in dets:
                    if d["id"] == selected_id:
                        target_det = d
                        break
            if target_det is None:
                target_det = pick_box(dets, intent["target"], intent["which"], W, H)
                selected_id = target_det["id"] if target_det else None
        
        yolo_view = get_yolo_view(base_frame.copy(), dets, intent, selected_id, target_det)
        
        # Get Depth view
        depth_view = depth_frame(base_frame)
        cv2.putText(depth_view, "DepthAnything", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Get Edge view
        edge_view = edge_frame(base_frame)
        cv2.putText(edge_view, "Edge Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Resize all views to same size for side-by-side display
        target_h, target_w = H, W
        yolo_resized = cv2.resize(yolo_view, (target_w, target_h))
        depth_resized = cv2.resize(depth_view, (target_w, target_h))
        edge_resized = cv2.resize(edge_view, (target_w, target_h))
        
        # Combine horizontally
        combined = np.hstack([yolo_resized, depth_resized, edge_resized])
        view = combined
        H, W = combined.shape[:2]

    if view is None:
        if frame is None:
            frame = get_frame()
            if frame is None:
                print("Camera frame not available. Exiting.")
                break
        H, W = frame.shape[:2]
        view = frame.copy()

    # Display current interaction mode and camera mode
    mode_names = {MODE_YOLO: "YOLO", MODE_DEPTH: "DepthAnything", MODE_EDGE: "Edge Detection"}
    if current_interaction_mode == INTERACTION_PERCEPTION:
        interaction_text = "Perception Mode (Press 'm' to stay, 'y' for zoom)"
    else:
        interaction_text = "YOLO Zoom Mode (Press 'y' to stay, 'm' for camera modes)"
    camera_mode_text = mode_names.get(current_mode, "Unknown") if current_mode != MODE_ALL else "All Views"
    
    if current_mode != MODE_ALL:
        mode_text = f"Camera: {camera_mode_text} | {interaction_text}"
        cv2.putText(view, mode_text, (10, H - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        # Add reminder about clicking window
        reminder_text = "Click this window to use keyboard shortcuts (m/y/1/2/3/4)"
        cv2.putText(view, reminder_text, (10, H - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    else:
        mode_text = f"Camera: {camera_mode_text} | {interaction_text}"
        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = (W - text_size[0]) // 2
        cv2.putText(view, mode_text, (text_x, H - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        # Add reminder about clicking window
        reminder_text = "Click this window to use keyboard shortcuts (m/y/1/2/3/4)"
        reminder_size = cv2.getTextSize(reminder_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        reminder_x = (W - reminder_size[0]) // 2
        cv2.putText(view, reminder_text, (reminder_x, H - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("LLM-Controlled Zoom (YOLO front-end)", view)
    
    # Handle key presses - 'm' and 'y' work in ALL modes to switch interaction modes
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC = quit
        break
    elif key == ord('m') or key == ord('M'):
        # Switch to perception change mode (works in any interaction mode)
        current_interaction_mode = INTERACTION_PERCEPTION
        print("✓ Switched to Perception Change Mode (Press 1/2/3/4 to change camera modes)")
        print("   Note: Click on the video window to use keyboard shortcuts")
    elif key == ord('y') or key == ord('Y'):
        # Switch to YOLO zoom mode (works in any interaction mode)
        current_interaction_mode = INTERACTION_YOLO_ZOOM
        print("✓ Switched to YOLO Zoom Mode (Type commands in terminal for zoom)")
        print("   Note: Click on the video window and press 'm' to switch back to camera modes")
    
    # Camera mode switching (only in perception mode)
    if current_interaction_mode == INTERACTION_PERCEPTION:
        if key == ord('1'):
            current_mode = MODE_YOLO
            intent = None
            selected_id = None
            print("✓ Switched to YOLO camera mode")
        elif key == ord('2'):
            current_mode = MODE_DEPTH
            intent = None
            selected_id = None
            print("✓ Switched to DepthAnything camera mode")
        elif key == ord('3'):
            current_mode = MODE_EDGE
            intent = None
            selected_id = None
            print("✓ Switched to Edge Detection camera mode")
        elif key == ord('4'):
            current_mode = MODE_ALL
            print("✓ Switched to All Views camera mode (showing YOLO, Depth, and Edge side by side)")

cap.release()
cv2.destroyAllWindows()
