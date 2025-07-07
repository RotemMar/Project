from pwi4_client import PWI4
import time
import cv2
import zwoasi as asi
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import os

# ========== Parameters ==========
EXPOSURE_TIME = 80
GAIN = 200
FOV_X_DEGREES = 0.5
FOV_Y_DEGREES = 0.3
threshold_deg = 0.01

# ========== Telescope Movement Functions ==========
def pixel_to_arcsec(x, y, FRAME_WIDTH, FRAME_HEIGHT):
    dx = x - FRAME_WIDTH // 2
    dy = y - FRAME_HEIGHT // 2
    angle_x_deg = dx / FRAME_WIDTH * FOV_X_DEGREES
    angle_y_deg = dy / FRAME_HEIGHT * FOV_Y_DEGREES
    return angle_x_deg * 3600, angle_y_deg * 3600

def wait_until_idle(pwi4, poll_interval=0.6):
    while pwi4.status().mount.is_slewing:
        time.sleep(poll_interval)

def get_altaz(pwi4):
    st = pwi4.status()
    return st.mount.altitude_degs, st.mount.azimuth_degs

def move(pwi4, direction, delta):
    alt, az = get_altaz(pwi4)
    if direction == "right":
        az += delta
    elif direction == "left":
        az -= delta
    elif direction == "up":
        alt += delta
    elif direction == "down":
        alt -= delta
    else:
        raise ValueError("Invalid direction")
    az %= 360.0
    alt = max(-10.0, min(alt, 90.0))
    pwi4.mount_goto_alt_az(alt, az)
    wait_until_idle(pwi4)

def control_telescope(pwi4, y, x, FRAME_WIDTH, FRAME_HEIGHT):
    arcsec_x, arcsec_y = pixel_to_arcsec(x, y, FRAME_WIDTH, FRAME_HEIGHT)
    deg_x = arcsec_x / 3600
    deg_y = arcsec_y / 3600
    if abs(deg_x) > threshold_deg:
        direction_x = "right" if deg_x > 0 else "left"
        move(pwi4, direction_x, abs(deg_x))
    if abs(deg_y) > threshold_deg:
        direction_y = "down" if deg_y > 0 else "up"
        move(pwi4, direction_y, abs(deg_y))

# ========== Camera Initialization ==========
sdk = r"C:\Users\rotem\OneDrive - mail.tau.ac.il\Uni\Forth Year\simester1\Project\camera\ASI_Camera_SDK\ASI_Camera_SDK\ASI_Windows_SDK_V1.37\ASI SDK\lib\x64\ASICamera2.dll"
asi.init(sdk)
num_cameras = asi.get_num_cameras()
if num_cameras == 0:
    raise ValueError("No cameras found")
camera = asi.Camera(0)
camera_info = camera.get_camera_property()
FRAME_WIDTH = camera_info['MaxWidth']
FRAME_HEIGHT = camera_info['MaxHeight']
camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, camera.get_controls()["BandWidth"]["MinValue"])
camera.disable_dark_subtract()
camera.auto_exposure()
camera.auto_wb()
camera.set_image_type(asi.ASI_IMG_RAW8)

# ========== YOLO Initialization ==========
model = YOLO("C:/Users/rotem/OneDrive - mail.tau.ac.il/Uni/Forth Year/simester1/Project/tel1/model_drone.pt")

# ========== Kalman Filter ==========
kf = cv2.KalmanFilter(6, 2)
kf.measurementMatrix = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
], dtype=np.float32)

dt=1
kf.transitionMatrix = np.array([
    [1, 0, dt, 0, 0.5*dt*dt, 0],
    [0, 1, 0, dt, 0, 0.5*dt*dt],
    [0, 0, 1, 0, dt, 0],
    [0, 0, 0, 1, 0, dt],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
], dtype=np.float32)


kf.processNoiseCov = np.diag([1, 1, 10, 10, 100, 100]).astype(np.float32) * 0.01
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5
kf.statePre = np.zeros((6, 1), dtype=np.float32)
kf.statePost = np.zeros((6, 1), dtype=np.float32)

# ========== Video Output ==========
os.makedirs("annotated_frames", exist_ok=True)
video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (FRAME_WIDTH, FRAME_HEIGHT))

# ========== Telescope Initialization ==========
pwi4 = PWI4()
s = pwi4.status()
if not s.mount.is_connected:
    pwi4.mount_connect()
if not s.mount.axis0.is_enabled:
    pwi4.mount_enable(0)
if not s.mount.axis1.is_enabled:
    pwi4.mount_enable(1)

# ========== Camera Warm-up ==========
camera.start_video_capture()
for _ in range(25):
    camera.capture_video_frame()
    time.sleep(0.1)

cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
prev_time = time.time()


# ========== Main Loop ==========
counter = 0
use_kalman = False
last_x = last_y = None

while True:
    now = time.time()
    #dt = now - prev_time
    dt = 0.4
    prev_time = now
    counter += 1
    if counter == 10:
        print("kalman")
        use_kalman = True

    filename = f"image.jpg"
    camera.capture_video_frame(filename=filename)
    image = cv2.imread(filename)
    time.sleep(0.05) #שינינוווווו

    results = model(image)
    annotated = results[0].plot()
    predicted = kf.predict()
    predicted_point = (int(predicted[0]), int(predicted[1]))
    cv2.circle(annotated, predicted_point, 10, (0, 0, 255), 2)

    boxes = results[0].boxes.xyxy
    if len(boxes) > 0:
        box = boxes[0].cpu().numpy()
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        kf.correct(measurement)
        cv2.circle(annotated, (int(center_x), int(center_y)), 10, (255, 0, 0), 2)

        if not use_kalman:
            if last_x is not None and last_y is not None:
                delta_x = center_x - last_x
                delta_y = center_y - last_y
                if abs(delta_x) > threshold_deg or abs(delta_y) > threshold_deg:
                    control_telescope(pwi4, center_y, center_x, FRAME_WIDTH, FRAME_HEIGHT)
            last_x, last_y = center_x, center_y

    if use_kalman:
        control_telescope(pwi4, predicted_point[1], predicted_point[0], FRAME_WIDTH, FRAME_HEIGHT)

    cv2.putText(annotated, "YOLO (measurement)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(annotated, "Kalman (prediction)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Tracking", annotated)
    cv2.imwrite(f"annotated_frames/frame_{counter:04d}.jpg", annotated)
    video_writer.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========== Cleanup ==========
cv2.destroyAllWindows()
camera.stop_video_capture()
camera.stop_exposure()
camera.close()
video_writer.release()
print("end")
