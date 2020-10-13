import face_recognition
import cv2
import numpy as np

def coding_img():
    """
    测试案例-保存一照片的编码数组：
    保存一个已知照片的编码数组到本地 txt 文件，测试能否在比较时成功
    Args:
        images_list (list[numpy.ndarray]): 传入一个 List，List 元素类型为 numpy.ndarray
    Returns:
        isMake(Boolean): 返回 Boolean 变量 isMask，Ture 表示戴口罩，False表示未带口罩
    """


    # 加载一个图片并学习如何识别它。
    yzc_image = face_recognition.load_image_file("../known_people/YZC.jpg")
    yzc_face_encoding = face_recognition.face_encodings(yzc_image)[0]

    # 加载一个示例图片并学习如何识别它。
    wuyj_image = face_recognition.load_image_file("../known_people/WuYongjun.jpg")
    wuyj_face_encoding = face_recognition.face_encodings(wuyj_image)[0]

    '''
    将编码数组保存到本地 txt 文件
    savez函数输出的是一个压缩文件(扩展名为npz)，其中每个文件都是一个save函数保存的npy文件，文件名对应于数组名。
    '''
    np.save("yzc_face_npy.npy", yzc_face_encoding)
    '''
    load函数自动识别npz文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容.
    '''
    yzc_face_npy = np.load("yzc_face_npy.npy")


    # print(yzc_face_npy)

    unknown_picture = face_recognition.load_image_file("../unknown_pictures/balck-yzc.jpg")
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

    # Now we can see the two face encodings are of the same person with `compare_faces`!

    results = face_recognition.compare_faces([yzc_face_npy], unknown_face_encoding, tolerance=0.5)




    # single img
    # unknown_picture = face_recognition.load_image_file("../unknown_pictures/balck-yzc.jpg")
    # unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
    #
    # results = face_recognition.compare_faces([all_npz['yzc_face_encoding']], unknown_face_encoding, tolerance=0.5)
    #
    # name = "Unknown"
    #
    if results[0] == True:
        print("It's a picture of me!")
    else:
        print("It's not a picture of me!")

coding()