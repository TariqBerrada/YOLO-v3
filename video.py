import tensorflow as tf
from utils import *
from yolov3 import YOLOv3Net
from convert_weights import load_weights
import cv2
import time

model_size = (416, 416,3)
num_classes = 80
class_name = './data/coco_names.txt'
max_output_size = 100
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5
cfgfile = 'cfg/yolov3_cfg.txt'
weightfile = 'weights/yolov3_weights.tf'

def main():
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    model.load_weights(weightfile)
    class_names = load_class_names(class_name)

    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    #To read from camera.
    #cap = cv2.VideoCapture(0)
    
    #To read a video file.
    videopath = 'data/videos/test.mp4'
    cap = cv2.VideoCapture(videopath)
    
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        i = 0
        while True:
            start = time.time()
            ret, frame = cap.read()
            #print(ret)
            if not ret:
                break
                
            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

            pred = model.predict(resized_frame)

            boxes, scores, classes, nums = output_boxes(pred, model_size, max_output_size = max_output_size, max_output_size_per_class = max_output_size_per_class, iou_threshold = iou_threshold, confidence_threshold = confidence_threshold)

            img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
            cv2.imshow(win_name, img)

            frame_dir = 'output/frames/frame_%d.jpg'%i
            cv2.imwrite(frame_dir, img)
            i += 1

            stop = time.time()

            elapsed_time = stop - start

            fps = int(1/elapsed_time)
            print("estimated fps : %d"%fps)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('Detection perfomed successfully.')

if __name__ == '__main__':
    main()