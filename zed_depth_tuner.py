import pyzed.sl as sl
import numpy as np
import cv2

# --------- Config ---------
Z_WIDTH  = 1280
Z_HEIGHT = 720
Z_FPS    = 30

# depth operating range (for clamping & viz), in meters
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 1.40

# initial visualization range in mm
VIS_MIN_M = 0.20
VIS_MAX_M = 1.40

# initial exposure / gain (0–100 in ZED SDK)
ZED_EXPOSURE_INIT = 40
ZED_GAIN_INIT     = 40

WINDOW_NAME = "ZED Depth Tuner"


def depth_stats_full(depth_mm: np.ndarray):
    valid = depth_mm[depth_mm > 0]
    total = depth_mm.size
    if valid.size == 0:
        return {
            "valid_ratio": 0.0,
            "min": None,
            "median": None,
            "max": None,
            "p5": None,
            "p95": None,
            "total": total,
            "n_valid": 0,
        }

    v = valid.astype(np.float32)
    return {
        "valid_ratio": float(valid.size) / float(total),
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "total": total,
        "n_valid": int(valid.size),
    }


def depth_to_colormap(depth_mm: np.ndarray, vis_min_mm: float, vis_max_mm: float):
    if depth_mm is None:
        return None
    valid = depth_mm > 0
    if not valid.any():
        norm = np.zeros_like(depth_mm, dtype=np.uint8)
    else:
        depth_f = depth_mm.astype(np.float32)
        norm = (depth_f - vis_min_mm) / max(1.0, (vis_max_mm - vis_min_mm))
        norm = np.clip(norm, 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)
        norm[~valid] = 0
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[depth_mm == 0] = 0
    return color


def safe_set_camera_setting(zed: sl.Camera, setting, value, label):
    try:
        err = zed.set_camera_settings(setting, int(value))
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[WARN] Failed to set {label}={value}: {err}")
    except Exception as e:
        print(f"[WARN] Exception while setting {label}={value}: {e}")


def main():
    # ---------- ZED setup ----------
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = Z_FPS
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = DEPTH_MIN_M
    init_params.depth_maximum_distance = DEPTH_MAX_M

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to open ZED: {err}")
        return

    # runtime params
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 80
    runtime.texture_confidence_threshold = 100
    runtime.enable_fill_mode = False  # start without aggressive fill

    image = sl.Mat()
    depth = sl.Mat()

    # manual exposure / gain
    exposure = ZED_EXPOSURE_INIT
    gain = ZED_GAIN_INIT
    safe_set_camera_setting(zed, sl.VIDEO_SETTINGS.AEC_AGC, 0, "AEC_AGC")
    safe_set_camera_setting(zed, sl.VIDEO_SETTINGS.EXPOSURE, exposure, "EXPOSURE")
    safe_set_camera_setting(zed, sl.VIDEO_SETTINGS.GAIN, gain, "GAIN")

    print("[INFO] ZED Depth Tuner controls:")
    print("  e/d : exposure +/-")
    print("  r/f : gain +/-")
    print("  1/2 : runtime.confidence_threshold +/- 10")
    print("  3/4 : runtime.texture_confidence_threshold +/- 10")
    print("  t   : print depth debug stats")
    print("  q   : quit")

    # ---------- UI setup ----------
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def on_trackbar(_):
        pass

    cv2.createTrackbar("min_depth_mm", WINDOW_NAME, int(VIS_MIN_M * 1000), 3000, on_trackbar)
    cv2.createTrackbar("max_depth_mm", WINDOW_NAME, int(VIS_MAX_M * 1000), 3000, on_trackbar)

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # meters

            # BGRA -> BGR
            zed_bgra = image.get_data()
            zed_bgr = cv2.cvtColor(zed_bgra, cv2.COLOR_BGRA2BGR)

            depth_m = depth.get_data().astype(np.float32)  # meters
            valid = np.isfinite(depth_m) & (depth_m > 0)
            depth_m[~valid] = 0.0

            # clamp to global depth min/max
            depth_m[(depth_m < DEPTH_MIN_M) | (depth_m > DEPTH_MAX_M)] = 0.0
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

            vis_min_mm = cv2.getTrackbarPos("min_depth_mm", WINDOW_NAME)
            vis_max_mm = cv2.getTrackbarPos("max_depth_mm", WINDOW_NAME)
            if vis_max_mm <= vis_min_mm:
                vis_max_mm = vis_min_mm + 1

            depth_vis = depth_to_colormap(depth_mm, vis_min_mm, vis_max_mm)

            # ensure same size (just in case)
            h, w = zed_bgr.shape[:2]
            if depth_vis.shape[:2] != (h, w):
                depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)

            vis = np.hstack([zed_bgr, depth_vis])

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # print depth stats
            if key == ord('t'):
                stats = depth_stats_full(depth_mm)
                print("========== ZED DEPTH DEBUG ==========")
                print(f"Exposure: {exposure} | Gain: {gain}")
                print(f"Confidence: {runtime.confidence_threshold} | Texture conf: {runtime.texture_confidence_threshold}")
                print(f"Valid pixels: {stats['n_valid']}/{stats['total']} ({stats['valid_ratio']*100:.1f}%)")
                if stats["min"] is None:
                    print("No valid depth.")
                else:
                    print(f"Depth mm: min={stats['min']:.1f}, median={stats['median']:.1f}, "
                          f"max={stats['max']:.1f}")
                    print(f"         p5={stats['p5']:.1f}, p95={stats['p95']:.1f}")
                print("=====================================")

            # exposure up/down
            if key == ord('e'):
                exposure = int(np.clip(exposure + 2, 0, 100))
                safe_set_camera_setting(zed, sl.VIDEO_SETTINGS.EXPOSURE, exposure, "EXPOSURE")
                print(f"[ZED] exposure = {exposure}")
            if key == ord('d'):
                exposure = int(np.clip(exposure - 2, 0, 100))
                safe_set_camera_setting(zed, sl.VIDEO_SETTINGS.EXPOSURE, exposure, "EXPOSURE")
                print(f"[ZED] exposure = {exposure}")

            # gain up/down
            if key == ord('r'):
                gain = int(np.clip(gain + 2, 0, 100))
                safe_set_camera_setting(zed, sl.VIDEO_SETTINGS.GAIN, gain, "GAIN")
                print(f"[ZED] gain = {gain}")
            if key == ord('f'):
                gain = int(np.clip(gain - 2, 0, 100))
                safe_set_camera_setting(zed, sl.VIDEO_SETTINGS.GAIN, gain, "GAIN")
                print(f"[ZED] gain = {gain}")

            # runtime confidence thresholds
            if key == ord('1'):
                runtime.confidence_threshold = int(np.clip(runtime.confidence_threshold + 10, 0, 100))
                print(f"[ZED] confidence_threshold = {runtime.confidence_threshold}")
            if key == ord('2'):
                runtime.confidence_threshold = int(np.clip(runtime.confidence_threshold - 10, 0, 100))
                print(f"[ZED] confidence_threshold = {runtime.confidence_threshold}")
            if key == ord('3'):
                runtime.texture_confidence_threshold = int(np.clip(runtime.texture_confidence_threshold + 10, 0, 100))
                print(f"[ZED] texture_confidence_threshold = {runtime.texture_confidence_threshold}")
            if key == ord('4'):
                runtime.texture_confidence_threshold = int(np.clip(runtime.texture_confidence_threshold - 10, 0, 100))
                print(f"[ZED] texture_confidence_threshold = {runtime.texture_confidence_threshold}")

    finally:
        zed.close()
        cv2.destroyAllWindows()
        print("[INFO] ZED closed, tuner done.")


if __name__ == "__main__":
    main()
