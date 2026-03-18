import time
import numpy as np
import cv2
import pyzed.sl as sl
from pupil_apriltags import Detector

FAMILY = "tag36h11"
TAG_SIZE_MM = 60
TAG_SIZE_M = TAG_SIZE_MM / 1000.0  # meters


def main():
    # -------------------
    # ZED init
    # -------------------
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NONE      # only RGB
    init.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED:", status)
        return

    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam

    fx = calib.fx
    fy = calib.fy
    cx = calib.cx
    cy = calib.cy

    print("=== ZED 2i LEFT camera ===")
    print(f"Resolution: {info.camera_configuration.resolution.width} x {info.camera_configuration.resolution.height}")
    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    print(f"AprilTag family: {FAMILY}, tag size: {TAG_SIZE_MM} mm")
    print("Press 'q' in the window to quit.\n")

    image_mat = sl.Mat()

    detector = Detector(
        families=FAMILY,
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False,
    )

    try:
        while True:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            # Get LEFT image (BGRA from ZED)
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            frame_bgra = image_mat.get_data()  # H x W x 4

            # Convert BGRA -> BGR for OpenCV
            frame_bgr = frame_bgra[:, :, :3].copy()

            # BGR -> gray for detector
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            results = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=TAG_SIZE_M,
            )

            # Draw detections
            if results:
                for r in results:
                    tid = r.tag_id
                    # corners: 4x2 array [ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]
                    corners = np.int32(r.corners)

                    # draw polygon around tag
                    cv2.polylines(frame_bgr, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

                    # draw center and ID text
                    cx_px, cy_px = r.center
                    cv2.circle(frame_bgr, (int(cx_px), int(cy_px)), 4, (0, 0, 255), -1)
                    text = f"ID {tid}"
                    cv2.putText(frame_bgr, text, (int(cx_px) + 5, int(cy_px) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            else:
                # optional: overlay hint
                cv2.putText(frame_bgr, "No tags detected",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("ZED AprilTag Viewer", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # tiny sleep so your CPU doesn't get roasted
            time.sleep(0.01)

    finally:
        zed.close()
        cv2.destroyAllWindows()
        print("Closed ZED and viewer.")

if __name__ == "__main__":
    main()
