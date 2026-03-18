import pyzed.sl as sl

def main():
    # Init params
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080  # or HD720/VGA
    init.depth_mode = sl.DEPTH_MODE.NONE           # we just want RGB for now
    init.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED:", status)
        return

    # Get camera info + intrinsics for LEFT eye
    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam

    print("=== LEFT camera intrinsics ===")
    print(f"fx = {calib.fx}")
    print(f"fy = {calib.fy}")
    print(f"cx = {calib.cx}")
    print(f"cy = {calib.cy}")
    print(f"Image size = {info.camera_configuration.resolution.width} x {info.camera_configuration.resolution.height}")

    # Grab one frame
    image = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)  # LEFT RGB
        frame = image.get_data()
        print("Frame shape:", frame.shape)       # (H, W, 4) BGRA
    else:
        print("Failed to grab image")

    zed.close()

if __name__ == "__main__":
    main()
