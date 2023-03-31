
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time


def main ():
    distance = '300cm_3'
    lighting = 'light_2'   
    # folders and files
    parent_folder = 'Data_Realsense'

    # check if folder exists
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    
    # check if folder exists
    pathh = os.path.join(parent_folder,distance, lighting)
    if not os.path.exists(pathh):
        os.makedirs(pathh, exist_ok=True)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    count = 0

    try:
        while count != 1:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

            # save images
            
            cv2.imwrite('color_frame.jpg', color_image)
            cv2.imwrite('depth_frame.jpg', depth_colormap)
            count = 1

            # Combine the RGB and depth data into a single array
            rgbd_data = np.concatenate((color_image.reshape(-1, 3), depth_image.reshape(-1, 1)), axis=1)

            # Save the RGB-D data as a PLY file
            with open('rgbd_data.ply', 'w') as f:
                # Write the PLY header
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write('element vertex {}\n'.format(rgbd_data.shape[0]))
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
                f.write('end_header\n')

                # Write the RGB-D data as ASCII values
                for i in range(rgbd_data.shape[0]):
                    r, g, b = color_image[i // 640, i % 640]
                    x, y, z = depth_image[i // 640, i % 640], depth_image[i // 640, i % 640], depth_image[i // 640, i % 640]
                    f.write('{:.3f} {:.3f} {:.3f} {} {} {}\n'.format(x, y, z, r, g, b))
            
            np.save('color_array.npy', color_image)
            np.save('depth_array.npy', depth_image)

    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    main()
