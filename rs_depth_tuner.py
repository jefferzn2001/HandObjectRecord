import pyrealsense2 as rs
import numpy as np
import cv2

# --------- Config ---------
WIDTH  = 848
HEIGHT = 480
FPS    = 30

# initial visualization depth range (meters)
VIS_MIN_M = 0.20
VIS_MAX_M = 1.40

# initial manual color exposure & gain (will get clamped)
COLOR_EXPOSURE = 100.0   # 50–200 usually good
COLOR_GAIN     = 50.0

WINDOW_NAME = "RealSense Depth Tuner"


def depth_stats_full(depth_mm: np.ndarray):
    """Return rich stats for debugging."""
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


def safe_set(sensor, option, value, label):
    try:
        sensor.set_option(option, value)
    except Exception as e:
        print(f"[WARN] Failed to set {label}={value}: {e}")


def main():
    global VIS_MIN_M, VIS_MAX_M, COLOR_EXPOSURE, COLOR_GAIN

    # ---------- RealSense setup ----------
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    device       = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    color_sensor = device.first_color_sensor()

    # ----- depth presets / emitter / laser -----
    preset_names = {
        rs.rs400_visual_preset.high_accuracy: "HIGH_ACCURACY",
        rs.rs400_visual_preset.high_density:  "HIGH_DENSITY",
        rs.rs400_visual_preset.default:       "DEFAULT",
    }
    current_preset = rs.rs400_visual_preset.high_accuracy

    def set_preset(preset):
        nonlocal current_preset
        current_preset = preset
        safe_set(depth_sensor, rs.option.visual_preset, preset, "visual_preset")
        print(f"[DEPTH] preset -> {preset_names.get(preset, str(preset))}")

    # initial preset
    set_preset(current_preset)

    emitter_supported = depth_sensor.supports(rs.option.emitter_enabled)
    laser_supported   = depth_sensor.supports(rs.option.laser_power)
    laser_range = depth_sensor.get_option_range(rs.option.laser_power) if laser_supported else None
    current_laser = None
    if laser_supported:
        current_laser = laser_range.max
        safe_set(depth_sensor, rs.option.laser_power, current_laser, "laser_power")
        print(f"[DEPTH] initial laser_power = {current_laser}")
    if emitter_supported:
        safe_set(depth_sensor, rs.option.emitter_enabled, 1.0, "emitter_enabled")
        print("[DEPTH] emitter_enabled -> 1")

    # optional: fix depth min/max range
    if depth_sensor.supports(rs.option.min_distance):
        safe_set(depth_sensor, rs.option.min_distance, 0.20, "min_distance")
    if depth_sensor.supports(rs.option.max_distance):
        safe_set(depth_sensor, rs.option.max_distance, 1.40, "max_distance")

    depth_scale = depth_sensor.get_depth_scale()  # meters per unit

    # get exposure/gain ranges so we never go out-of-range
    exp_range  = color_sensor.get_option_range(rs.option.exposure) if color_sensor.supports(rs.option.exposure) else None
    gain_range = color_sensor.get_option_range(rs.option.gain) if color_sensor.supports(rs.option.gain) else None

    if exp_range:
        COLOR_EXPOSURE = float(np.clip(COLOR_EXPOSURE, exp_range.min, exp_range.max))
    if gain_range:
        COLOR_GAIN = float(np.clip(COLOR_GAIN, gain_range.min, gain_range.max))

    # color: manual exposure / gain
    if color_sensor.supports(rs.option.enable_auto_exposure):
        safe_set(color_sensor, rs.option.enable_auto_exposure, 0.0, "color_auto_exposure")
    if exp_range:
        safe_set(color_sensor, rs.option.exposure, COLOR_EXPOSURE, "color_exposure")
    if gain_range:
        safe_set(color_sensor, rs.option.gain, COLOR_GAIN, "color_gain")

    print("[INFO] Keys:")
    print("  e/d : exposure +/-")
    print("  r/f : gain +/-")
    print("  1/2/3 : depth preset (HIGH_ACCURACY / HIGH_DENSITY / DEFAULT)")
    print("  z : toggle emitter on/off")
    print("  c/v : laser power +/-")
    print("  t : print depth debug stats")
    print("  q : quit")
    if exp_range:
        print(f"[INFO] Exposure range: [{exp_range.min}, {exp_range.max}] (current {COLOR_EXPOSURE})")
    if gain_range:
        print(f"[INFO] Gain range:     [{gain_range.min}, {gain_range.max}] (current {COLOR_GAIN})")
    if laser_range:
        print(f"[INFO] Laser range:    [{laser_range.min}, {laser_range.max}] (current {current_laser})")

    # ---------- UI setup ----------
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def on_trackbar(_):
        pass

    cv2.createTrackbar("min_depth_mm", WINDOW_NAME, int(VIS_MIN_M * 1000), 3000, on_trackbar)
    cv2.createTrackbar("max_depth_mm", WINDOW_NAME, int(VIS_MAX_M * 1000), 3000, on_trackbar)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())          # RGB
            depth_raw = np.asanyarray(depth_frame.get_data())      # uint16 (device units)

            # convert depth to mm and clamp to a reasonable range
            depth_m  = depth_raw.astype(np.float32) * depth_scale
            depth_m[(depth_m < 0.1) | (depth_m > 3.0)] = 0.0
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

            # Rotate images 180 degrees (camera mounted upside down)
            depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_180)
            color = cv2.rotate(color, cv2.ROTATE_180)

            # read viz range from trackbars
            vis_min_mm = cv2.getTrackbarPos("min_depth_mm", WINDOW_NAME)
            vis_max_mm = cv2.getTrackbarPos("max_depth_mm", WINDOW_NAME)
            if vis_max_mm <= vis_min_mm:
                vis_max_mm = vis_min_mm + 1

            depth_vis = depth_to_colormap(depth_mm, vis_min_mm, vis_max_mm)
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            vis = np.hstack([color_bgr, depth_vis])

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # print depth stats + current exposure/gain
            if key == ord('t'):
                stats = depth_stats_full(depth_mm)
                print("========== DEPTH DEBUG ==========")
                print(f"Exposure: {COLOR_EXPOSURE} | Gain: {COLOR_GAIN}")
                if laser_range:
                    print(f"Laser: {current_laser}")
                if emitter_supported:
                    try:
                        cur_em = depth_sensor.get_option(rs.option.emitter_enabled)
                    except Exception:
                        cur_em = None
                    print(f"Emitter: {cur_em}")
                print(f"Valid pixels: {stats['n_valid']}/{stats['total']} "
                      f"({stats['valid_ratio']*100:.1f}%)")
                if stats["min"] is None:
                    print("No valid depth.")
                else:
                    print(f"Depth mm: min={stats['min']:.1f}, median={stats['median']:.1f}, "
                          f"max={stats['max']:.1f}")
                    print(f"         p5={stats['p5']:.1f}, p95={stats['p95']:.1f}")
                print("=================================")

            # exposure up (e), down (d)
            if key == ord('e') and exp_range:
                COLOR_EXPOSURE = float(np.clip(COLOR_EXPOSURE + 10, exp_range.min, exp_range.max))
                safe_set(color_sensor, rs.option.exposure, COLOR_EXPOSURE, "color_exposure")
                print(f"[COLOR] exposure = {COLOR_EXPOSURE}")
            if key == ord('d') and exp_range:
                COLOR_EXPOSURE = float(np.clip(COLOR_EXPOSURE - 10, exp_range.min, exp_range.max))
                safe_set(color_sensor, rs.option.exposure, COLOR_EXPOSURE, "color_exposure")
                print(f"[COLOR] exposure = {COLOR_EXPOSURE}")

            # gain up (r), down (f)
            if key == ord('r') and gain_range:
                COLOR_GAIN = float(np.clip(COLOR_GAIN + 5, gain_range.min, gain_range.max))
                safe_set(color_sensor, rs.option.gain, COLOR_GAIN, "color_gain")
                print(f"[COLOR] gain = {COLOR_GAIN}")
            if key == ord('f') and gain_range:
                COLOR_GAIN = float(np.clip(COLOR_GAIN - 5, gain_range.min, gain_range.max))
                safe_set(color_sensor, rs.option.gain, COLOR_GAIN, "color_gain")
                print(f"[COLOR] gain = {COLOR_GAIN}")

            # switch depth presets
            if key == ord('1'):
                set_preset(rs.rs400_visual_preset.high_accuracy)
            if key == ord('2'):
                set_preset(rs.rs400_visual_preset.high_density)
            if key == ord('3'):
                set_preset(rs.rs400_visual_preset.default)

            # toggle emitter
            if key == ord('z') and emitter_supported:
                try:
                    cur = depth_sensor.get_option(rs.option.emitter_enabled)
                except Exception:
                    cur = 1.0
                new_val = 0.0 if cur > 0.5 else 1.0
                safe_set(depth_sensor, rs.option.emitter_enabled, new_val, "emitter_enabled")
                print(f"[DEPTH] emitter_enabled -> {new_val}")

            # laser power up/down
            if key == ord('c') and laser_supported:
                current_laser = float(np.clip(current_laser + 5.0, laser_range.min, laser_range.max))
                safe_set(depth_sensor, rs.option.laser_power, current_laser, "laser_power")
                print(f"[DEPTH] laser_power = {current_laser}")
            if key == ord('v') and laser_supported:
                current_laser = float(np.clip(current_laser - 5.0, laser_range.min, laser_range.max))
                safe_set(depth_sensor, rs.option.laser_power, current_laser, "laser_power")
                print(f"[DEPTH] laser_power = {current_laser}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stopped RealSense, depth tuner closed.")


if __name__ == "__main__":
    main()
