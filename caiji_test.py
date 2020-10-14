from util import encoding_face,save_face_encoding
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
    flag = True
    if flag:
        code_face = encoding_face.get_coding_img(rgb_small_frame=rgb_small_frame)
        save_face_encoding.save_encoding(name_str="不是陌生人",face_encode=code_face)
        print("SUWEI========================")
        flag = False

    # 只需每隔一帧处理一次即可节省时间
    if process_this_frame:
        # 通过 `data` 传入 image 对象
        # input_dict = {"data": [cv2.imread(rgb_small_frame)]}
        # results = module.face_detection(data=input_dict)
        results = module.face_detection(images=imglist)
        print(results)

    process_this_frame = not process_this_frame

    # 显示结果图像
    cv2.imshow('Video', frame)

    # 按键盘上的“q”退出！
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close 摄像头 handle
video_capture.release()
cv2.destroyAllWindows()

