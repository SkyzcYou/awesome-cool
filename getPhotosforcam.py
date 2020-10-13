import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)


process_this_frame = True

while True:
    # 抓取一帧视频
    ret, frame = video_capture.read()

    print(frame)

    # 显示结果图像
    cv2.imshow('Video', frame)

    # 按键盘上的“q”退出！
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close 摄像头 handle
video_capture.release()
cv2.destroyAllWindows()
