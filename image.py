import tensorflow as tf
from utils import *
import cv2
import numpy as np
from yolov3 import YOLOv3Net
from convert_weights import load_weights
#physical_devices = tf.config.experimental.list_physical_devices('GPU')

#assert len(physical_devices) > 0, 'No GPU available, running on CPU.'
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_size = (416, 416, 3)
num_classes = 80
class_name = './data/coco_names.txt'
max_output_size = 40
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

weightfile = "weights/yolov3.weights"
cfgfile = "cfg/yolov3_cfg.txt"
img_path = 'data/images/test.jpg'

def main():

    model = YOLOv3Net(cfgfile, model_size, num_classes)
    load_weights(model, cfgfile, weightfile)
    class_names = load_class_names(class_name)

    image = cv2.imread(img_path)
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    resized_frame = resize_image(image, (model_size[0], model_size[1]))
    pred = model.predict(resized_frame)

    boxes, scores, classes, nums = output_boxes(pred, model_size, max_output_size = max_output_size, max_output_size_per_class = max_output_size_per_class, iou_threshold = iou_threshold, confidence_threshold = confidence_threshold)

    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)

    win_name = 'Image detection'
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()