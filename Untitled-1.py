import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

def find_blue_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), largest_contour
    return None, None

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open the video fil")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    center_frame_x = frame_width / 2
    center_frame_y = frame_height / 2

    center_xs = []
    center_ys = []
    distances = []
    timestamps = []

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        center, contour = find_blue_center(frame)
        if center:
            center_x, center_y = center
            center_xs.append(center_x)
            center_ys.append(center_y)
            distance = np.sqrt((center_x - center_frame_x) ** 2 + (center_y - center_frame_y) ** 2)
            distances.append(distance)
        else:
            center_xs.append(None)
            center_ys.append(None)
            distances.append(None)

        timestamps.append(frame_number / fps)
        frame_number += 1

        if center:
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

        cv2.circle(frame, (int(center_frame_x), int(center_frame_y)), 5, (0, 255, 0), -1)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    # Plot: Distance from center as a function of time
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, distances)  # color='#800080'purple 
    plt.xlabel("Time (seconds)")
    plt.ylabel("Distance (pixels)")
    plt.title("Distance of the drone from the center over time (video4)")
    plt.grid()
    plt.tight_layout()
    plt.show()


# <<< Replace file path >>>
process_video("C:/Users/rotem/OneDrive - mail.tau.ac.il/Uni/Forth Year/simester1/Project/graphs/video4.mp4")
