import cv2
import numpy as np
from ultralytics import YOLO

# config values
VIDEO_PATH = "data/play.mp4"
CONFIDENCE_THRESHOLD = 0.5
VELOCITY_THRESHOLD = 2.5

SIDELINE_MARGIN = 0.05     # 5% from sides
ENDZONE_MARGIN = 0.15      # top/bottom zone

OUT_OF_BOUNDS_FRAMES = 10
TOUCHDOWN_FRAMES = 15
EVENT_DISPLAY_TIME = 60

# load yolo model
model = YOLO("yolov8n.pt")

# video init
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

prev_gray = None
frame_count = 0
snap_frame = None
snap_detected = False

player_history = {}

# counters for events
oob_counter = 0
td_counter = 0

oob_confirmed = False
td_confirmed = False

oob_timer = 0
td_timer = 0

# main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    height, width, _ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect sudden motion (snap)
    if prev_gray is not None and not snap_detected:
        diff = cv2.absdiff(prev_gray, gray)
        motion_score = np.sum(diff)

        if motion_score > 5e6:
            snap_frame = frame_count
            snap_detected = True

    prev_gray = gray

    # run object detection
    results = model(frame, verbose=False)[0]
    players = []

    # filter players only
    for box in results.boxes:
        if int(box.cls[0]) == 0 and box.conf[0] > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            players.append((x1, y1, x2, y2, cx, cy))

    # estimate line of scrimmage from player positions
    los_x = None
    if players:
        los_x = int(np.mean([p[4] for p in players]))
        cv2.line(frame, (los_x, 0), (los_x, height), (0, 255, 0), 2)

    out_of_bounds_detected = False
    touchdown_candidate = False

    # track players + check events
    for i, (x1, y1, x2, y2, cx, cy) in enumerate(players):

        if i not in player_history:
            player_history[i] = []

        player_history[i].append(cx)

        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # sideline check
        if cx < width * SIDELINE_MARGIN or cx > width * (1 - SIDELINE_MARGIN):
            out_of_bounds_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # simple offside check around snap
        if snap_detected and len(player_history[i]) >= 2 and los_x is not None:
            prev_x = player_history[i][-2]
            curr_x = player_history[i][-1]
            velocity = abs(curr_x - prev_x)

            crossed = prev_x < los_x and curr_x > los_x

            if crossed and velocity > VELOCITY_THRESHOLD and frame_count < snap_frame:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # endzone check
        if cy < height * ENDZONE_MARGIN or cy > height * (1 - ENDZONE_MARGIN):
            touchdown_candidate = True

    # confirm out of bounds over multiple frames
    if out_of_bounds_detected:
        oob_counter += 1
    else:
        oob_counter = 0

    if oob_counter >= OUT_OF_BOUNDS_FRAMES:
        oob_confirmed = True
        oob_timer = EVENT_DISPLAY_TIME

    # confirm touchdown over multiple frames
    if touchdown_candidate:
        td_counter += 1
    else:
        td_counter = 0

    if td_counter >= TOUCHDOWN_FRAMES:
        td_confirmed = True
        td_timer = EVENT_DISPLAY_TIME

    # handle display timers
    if oob_timer > 0:
        oob_timer -= 1
    else:
        oob_confirmed = False

    if td_timer > 0:
        td_timer -= 1
    else:
        td_confirmed = False

    # draw event text
    y = 30

    if snap_detected:
        cv2.putText(frame, "EVENT: SNAP", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        y += 30

    if oob_confirmed:
        cv2.putText(frame, "EVENT: OUT OF BOUNDS", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        y += 30

    if td_confirmed:
        cv2.putText(frame, "EVENT: TOUCHDOWN CANDIDATE", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

    # show frame
    cv2.imshow("AI NFL Referee (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
