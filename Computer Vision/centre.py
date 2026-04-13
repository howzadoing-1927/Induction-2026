import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/sharadkumaraggarwal/PycharmProjects/OpenCV learning/.venv/TargetVideo.mp4')
lower_blue = np.array([150,100,20])
upper_blue = np.array([230,200,130])

filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = 25.0
frame_size = (540,380)

out = cv2.VideoWriter(filename, fourcc, fps, frame_size)



frame_count = 0
while (cap.isOpened()):

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (540, 380), fx=0, fy=0,
                       interpolation=cv2.INTER_CUBIC)
    mask = cv2.inRange(frame, lower_blue, upper_blue)


    ccnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(ccnt) > 0:
        rim_cnt = max(ccnt, key=cv2.contourArea)

    M = cv2.moments(rim_cnt)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0


    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
    cv2.imshow('Frame', frame)
    out.write(frame)

    if frame_count % 30 == 0:
         print(f"Processed {frame_count} frames...")
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
