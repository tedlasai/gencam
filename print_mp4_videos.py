import os
import cv2

import os
import cv2



def print_mp4_video_shapes(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                file_path = os.path.join(root, filename)

                # File size
                size_mb = os.path.getsize(file_path) / (1024 * 1024)

                # Open video
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    print(f"{file_path}: Could not open video.")
                    continue

                # Get metadata
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Try to get channels from the first frame
                success, frame = cap.read()
                if success and frame is not None:
                    channels = frame.shape[2] if len(frame.shape) == 3 else 1
                else:
                    channels = "?"
                cap.release()

                shape = (frame_count, height, width, channels)
                print(f"{file_path} — {size_mb:.2f} MB — Shape: {shape}")

# Example usage:
print_mp4_video_shapes("/home/tedlasai/genCamera/GoProResults/Favaro")