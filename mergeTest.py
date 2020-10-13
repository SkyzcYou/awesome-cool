import face_recognition
import cv2
import numpy as np
import paddlehub as hub
from util import detect_mask

video_capture = cv2.VideoCapture(0)

# 加载一个示例图片并学习如何识别它。
yzc_image = face_recognition.load_image_file("known_people/YZC.jpg")
yzc_face_encoding = face_recognition.face_encodings(yzc_image)[0]

# 加载一个示例图片并学习如何识别它。
jack_image = face_recognition.load_image_file("known_people/jack.jpg")
jack_face_encoding = face_recognition.face_encodings(jack_image)[0]

# 加载一个示例图片并学习如何识别它。
wuyj_image = face_recognition.load_image_file("known_people/WuYongjun.jpg")
wuyj_face_encoding = face_recognition.face_encodings(jack_image)[0]

# 创建已知人脸编码及其名称的数组
known_face_encodings = [
    yzc_face_encoding,
    jack_face_encoding,
    wuyj_face_encoding
]
known_face_names = [
    "Youzhengcai",
    "Jack Ma",
    "WuYongjun"
]

# 初始化变量
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

module = hub.Module(name="pyramidbox_lite_server_mask")


while True:
    # 抓取一帧视频
    ret, frame = video_capture.read()

    # 将视频帧调整为1/4大小，以便更快地进行人脸识别处理
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    imglist = [small_frame]
    # 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
    rgb_small_frame = small_frame[:, :, ::-1]

    # 只需每隔一帧处理一次即可节省时间
    if process_this_frame:
        # 查找当前视频帧中的所有人脸和人脸编码
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 查看该面是否与已知面匹配
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
            name = "Unknown"

            # 如果在已知的面编码中找到匹配项，只需使用第一个。
            # 如果匹配项中为True：
            # 第一个匹配索引=匹配.索引（正确）
            # 名称=已知的面名称[第一个匹配的索引]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # 显示结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 缩小面部位置，因为我们检测到的帧被缩放到1/4大小
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 在面下方绘制一个名称的标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 显示结果图像
    cv2.imshow('Video', frame)

    ''' 直接调用 '''
    # results = module.face_detection(images=imglist)
    # print(results)
    ''' 工具方法 detect_mask() 调用 '''
    results = detect_mask.detect_mask(imglist)
    print(results)

    # 按键盘上的“q”退出！
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close 摄像头 handle
video_capture.release()
cv2.destroyAllWindows()
