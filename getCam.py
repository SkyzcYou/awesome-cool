import paddlehub as hub
import cv2

video_capture = cv2.VideoCapture(0)
process_this_frame = True

module = hub.Module(name="pyramidbox_lite_server_mask")

while True:
    # 抓取一帧视频
    ret, frame = video_capture.read()

    # 将视频帧调整为1/4大小，以便更快地进行人脸识别处理
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
    imglist = [small_frame]
    # 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
    rgb_small_frame = small_frame[:, :, ::-1]
    # # (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
    # imglist = [rgb_small_frame]

    # 只需每隔一帧处理一次即可节省时间
    if process_this_frame:


        # 通过 `data` 传入 image 对象
        # input_dict = {"data": [cv2.imread(rgb_small_frame)]}
        # results = module.face_detection(data=input_dict)
        results = module.face_detection(images=imglist)
        print(results)

    process_this_frame = not process_this_frame

    # 显示结果
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

    # 显示结果图像
    cv2.imshow('Video', frame)

    # 按键盘上的“q”退出！
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close 摄像头 handle
video_capture.release()
cv2.destroyAllWindows()
