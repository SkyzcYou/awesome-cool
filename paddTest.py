import paddlehub as hub
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1.待预测图片
test_img_path = ["/home/giftiay/PycharmProjects/awesome-cool/detection_result/test.jpg"]

# 2.载入模型
module = hub.Module(name="pyramidbox_lite_mobile_mask")

# 3.预测
input_dict = {"image": test_img_path}
results = module.face_detection(data=input_dict)

# 4.结果展示
img = mpimg.imread("detection_result/test.jpg")
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()