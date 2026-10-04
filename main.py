import cv2
from pathlib import Path

video = next(Path(r"D:\BuiGiaHuy\CN8\EXE202\dataset\train\Ăn").glob("*.mp4"))

cap = cv2.VideoCapture(str(video))

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = frames / fps if fps else 0

print("Video:", video.name)
print("FPS:", fps)
print("Frames:", frames)
print("Resolution:", width, "x", height)
print("Duration:", duration, "seconds")

cap.release()