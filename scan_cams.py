# scan_cams.py
import cv2

def try_open(idx, api):
    cap = cv2.VideoCapture(idx, api)
    ok = cap.isOpened()
    if ok:
        ok, frame = cap.read()
        cap.release()
        return ok
    return False

apis = [cv2.CAP_AVFOUNDATION]  # macOS native
labels = {cv2.CAP_AVFOUNDATION: "AVFOUNDATION"}

print("Scanning camera indexes (0..5) with AVFOUNDATION...")
working = []
for i in range(6):
    ok = try_open(i, cv2.CAP_AVFOUNDATION)
    print(f"  index {i}: {'OK' if ok else 'fail'}")
    if ok:
        working.append(i)

if not working:
    print("\nNo camera opened. Try:")
    print("  • Ensure Terminal/IDE has Camera permission")
    print("  • Close apps using the camera")
    print("  • Re-run with different index after granting permission")
else:
    print(f"\nWorking indexes: {working}")
