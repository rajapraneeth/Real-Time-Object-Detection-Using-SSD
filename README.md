# Real-Time-Object-Detection-Using-SSD
Using pre-trained MobileNet SSD for Real Time Multi-Class-Object Detection
There are two type of deep neural networks here.

Base network
detection network
MobileNet, VGG-Net, LeNet and all of them are base networks. Base network provide high level features for classification or detection. For classification we add a fully connected layer at the end of this networks. But if we remove fully connected layer and replace it with detection networks, like SSD, Faster R-CNN, and so on. In fact, SSD use of last convolutional layer on base networks for detection task. MobileNet just like other base networks use of convolution to produce high level features.

This module loads pre-trained model for multiclass object detection from a video feed. Besides MobileNet-SDD other architectures can also be used for the same.

GoogleLeNet
YOLO
SqueezeNet
Faster R-CNN
ResNet
Installation
git clone https://github.com/rmundra22/Real-Time-Object-Detection.git
cd Real-Time-Object-Detection
Create environment (MacOS) and install requirements
python --version (Check you python versiona and make changes accordingly, say 3.x => 3.9)
virtualenv -p python3.x obj_det
source ./obj_det/bin/activate
pip install -r requiremnets.txt
Usage of the file
The above code establish the following arguments:

video: Path file video.
prototxt: Network file is .prototxt
weights: Network weights file is .caffemodel
thr: Confidence threshold.
Runnning this file

Download the pretrained model files namely 'MobileNetSSD_deploy.prototxt' and 'MobileNetSSD_deploy.caffemodel' files.
Check if the video camera in your device works properly. Code switches it on automatically once the code starts.
Use the below commond to execute the python file:- python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel (if this doesn't work, try giving absolute paths or set project directory)
My Experiments
Sample - Experiment_1

In the above gif we can observe that we are able to detect multiple objects in real-time using laptop's video cam. But it may happen sometimes that the model fail to detect an object class or it may also happen that occuluded objects are neither detected/correctly-classified. It's a pretty basic model to give a good feel of object detection without exclusively training a model with some personilized dataset. Retraining the model with some personalized data may help to give better results.

Sample - Experiment_2

Single Shot Detection (SSD)
In simple words it means you take a single look at an image to propose object-detections even if we are require to detect multiple objects within the image. The difference between the SSD and Regional Proposal Network (RPN) based approaches such as R-CNN series is that they are a 2 stage algorithms. In other words it is that they need two shots, one for generating region proposals and another one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches. Because of this feature of the SSD based approach we can get real time detections.

SSD300 achieves 74.3% mAP at 59 FPS while SSD500 achieves 76.9% mAP at 22 FPS, which outperforms Faster R-CNN (73.2% mAP at 7 FPS) and YOLOv1 (63.4% mAP at 45 FPS). This Single Shot Multibox Detector is Cited by 6083 when I am writing this. Some Crazy Object Detection Results on MS COCO dataset can be visualize using the image below :-

Crazy Object Detections
