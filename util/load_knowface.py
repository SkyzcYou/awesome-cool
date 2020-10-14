import face_recognition

def load_face_data():
    # 已知人脸库 Start
    # 加载一个示例图片并学习如何识别它。
    yzc_image = face_recognition.batch_face_locations("/know_face_img/YZC.jpg")
    yzc_face_encoding = face_recognition.face_encodings(yzc_image)[0]

    # 加载一个示例图片并学习如何识别它。

    jack_image = face_recognition.load_image_file("/know_face_img/jack.jpg")
    jack_face_encoding = face_recognition.face_encodings(jack_image)[0]

    # 加载一个示例图片并学习如何识别它。
    wuyj_image = face_recognition.load_image_file("/know_face_img/WuYongjun.jpg")
    wuyj_face_encoding = face_recognition.face_encodings(wuyj_image)[0]

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
    # 知人脸库 End

load_face_data()