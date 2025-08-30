import pyrealsense2 as rs
import numpy as np
import cv2

# Path to your .bag file (downloaded dataset or recorded yourself)
bag_file = "walking.bag"

# Configure pipeline for playback
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)  # False = stop when file ends

# Enable depth and color streams
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start pipeline
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            break

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Colorize depth for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                           cv2.COLORMAP_JET)

        # Show both depth and color
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow("RealSense Playback", images)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
