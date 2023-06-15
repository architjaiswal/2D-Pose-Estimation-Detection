# What A Pose: 2D pose estimation models

A 2D pose estimation model can be used in various applications, such as sports analysis, rehabilitation, and human-
computer interaction. For example, the model can be used to track the gesture of a human, allowing the autonomous
robot to perform a task corresponding to the operator’s gesture. Similarly, the model can be used in rehabilitation to
monitor patient progress and ensure they perform exercises correctly. This project aims to contribute to developing
a robust and accurate 2D pose estimation using Machine Learning (ML) models to detect and track human body
poses from 2D images. ML models like logistic regression, CNN, and MoveNet were deployed and evaluated on color
image datasets of various Yoga poses. Additional techniques, such as data augmentation and transfer learning, were
used to improve the model’s accuracy. The ML model predicts the location of joints and body parts in the image and
classifies yoga poses. Performance evaluation was performed using various metrics such as accuracy, precision, recall,
and F1-score. The results of the performance evaluation of all the ML models trained in this project and pose detection
accuracy are discussed in this paper.

## [Project Video](https://youtu.be/GgoXQ3LO-6s)

## Code Implementation

final_project.ipynb contains the implementation of various models trained to perform pose detection on 2D images. It also requires several libraries which are imported in the notebook.

Dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset?select=DATASET)
Keep the dataset in same format, it should have TRAIN and TEST directory. Each should contain multiple sub-directories named with a Yoga pose.

## Training Data
Retraining the models will require you to have the pretrained weights file for MoveNet. The pretrained weights file was downloaded from TensorFlow's official website and its name is "movenet_thunder.tflite".
Training all the models will also need the complete dataset and data must be kept in the same format as available on the link above. 

## Testing
Testing the models will also require the test images to be in the same directory and label format as shown in the Kaggle link above. Any test image can be passed through the MoveNet model, but in order to perform the evaluation (like accuracy, confusion matrix, etc.) it needs to have the image or the directory labeled in the right format.

IMPORTANT: The present code requires a GPU device to execute several lines. If you get an error when the code is trying to access a GPU, then consider commenting those lines and reindent the code. 

## Dependencies

- cv2
- tensorflow
- tqdm
- skimage
- keras
- utils.py
- data.py
- movenet.py

## Author

Archit. K. Jaiswal
