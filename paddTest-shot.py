import paddlehub as hub
import cv2
import json

module = hub.Module(name="pyramidbox_lite_server_mask")

# 通过 `data` 传入 image 对象
# input_dict = {"data": [cv2.imread("/home/giftiay/PycharmProjects/awesome-cool/known_people/pony.jpg")]}
# results = module.face_detection(data=input_dict)

results = module.face_detection(images=[cv2.imread('/home/giftiay/PycharmProjects/awesome-cool/known_people/pony.jpg')])
print(results)