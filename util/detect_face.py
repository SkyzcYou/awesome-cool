import face_recognition
import cv2
import numpy as np

def get_detect_face(rgb_frame):

    """
    人脸检测及人脸识别
    检测图片中的人脸位置、识别人脸数据
    Args:
        rgb_frame (numpy.ndarray): 传入一个经过 RGB 处理后的 OpenCV 格式的图像
    Returns:
        name : 返回图片中人物的姓名
    """



    # 初始化变量
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True


    # 对传入的 RGB 处理后的 OpenCV 格式的图像进行编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)[0]


    # 查看该面是否与已知面匹配
    matches = face_recognition.compare_faces(known_face_encodings, face_encodings, tolerance=0.4)
    name = "Unknown"

    # 如果在已知的面编码中找到匹配项，只需使用第一个。
    # 如果匹配项中为True：
    # 第一个匹配索引=匹配.索引（正确）
    # 名称=已知的面名称[第一个匹配的索引]

    # 或者，使用与新面的距离最小的已知面
    face_distances = face_recognition.face_distance(known_face_encodings, face_encodings)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)
    print(name)
    return name


        # # 显示结果
        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #     top *= 4
        #     right *= 4
        #     bottom *= 4
        #     left *= 4
        #
        #     # 缩小面部位置，因为我们检测到的帧被缩放到1/4大小
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #
        #     # 在面下方绘制一个名称的标签
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        #
        # # 显示结果图像
        # cv2.imshow('Video', frame)
        #
        # # 按键盘上的“q”退出！
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
