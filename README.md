# YOLO-v3
An Implementation of a pre-trained YOLO network (version 3) for object detection on tensorflow 2.0.
## About the model
The architecture of the network and its relevant parameters are present in the cfg file, the prediction is done on 81 labeled classes with pre-training on the COCO dataset.

The model is generated by creating its architecture first using the cfg file then loading the pretrained weights into it.

Once the trained model is imported, inference is made and we make use of thresholding and non max supression functions in order to keep only relevant boxes that are detected using the state space output, we can then plot these detections or return the coordinates for another algorithm to work with.
## Code Structure
- **yolov3.py** : contains the model class, generates model structure and loads pre-trained weights to the graph.
- **utils.py** : contains functions for image processing, thresholding and max supression for detecting unique boxes with the highest confidence scores.
- **convert_weights.py** : used to convert weights file which contains the models weights as a float list into a format usable by tensorflow.
- **image** : example of prediction on an image file.
- **video.py** : example of prediction on a video file.
## Result
![](result.gif)
