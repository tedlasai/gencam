import cv2
import numpy as np

# Define output video settings
output_filename = 'videoio_test.mp4'
frame_width = 640
frame_height = 480
fps = 30

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 'avc1' = H.264 codec (alternative: 'H264' or 'X264')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Check if VideoWriter opened successfully
if not out.isOpened():
    print("❌ Failed to open VideoWriter with H.264 codec.")
else:
    print("✅ VideoWriter opened successfully. Writing frames...")

    # Write 60 frames (2 seconds of video at 30fps)
    for _ in range(60):
        frame = np.random.randint(0, 256, (frame_height, frame_width, 3), dtype=np.uint8)  # Random colorful noise
        out.write(frame)

    out.release()
    print(f"✅ Video writing done! Saved as {output_filename}")
