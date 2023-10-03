# DL-for-dogs-activity-recognition
Hadas Manor and Hadar Shloosh's project about dogs' activity recognition using DL  
* photo  
# Introduction 
Adopting dogs from animal shelters plays a crucial role in addressing multiple societal and ethical concerns.
It aids in curbing the issue of pet overpopulation by providing homes to dogs that might otherwise contribute to the strain on shelter resources or face the risk of euthanasia due to lack of space.
Despite the many successful instances of matching dogs with families, there exist instances where the compatibility between a dog and its adopted family proves challenging.These mismatches can lead to stress for both the family and the dog, potentially resulting in the dog being returned to the shelter. Data from animal shelters reveal that a significant percentage of returned dogs are due to issues related to behavior, health concerns, or simply a mismatch in lifestyle or energy levels between the dog and the adopting family.

Studies have shown that depressed dogs often experience longer stays in shelters, reduced chances of adoption, and increased susceptibility to health issues. This underscores the urgency of addressing the emotional well-being of shelter dogs through enriching activities, socialization programs, and personalized care.

# Project goals 
In our project, we aim to predict dogs' behavior through videos taken at dogs' shelter. This prediction serves two main purposes. Firstly, it provides a valuable tool for the shelter to identify dogs that may be experiencing depression or require additional attention and medical assistance. By monitoring their activity levels and behavior, we can better assess their well-being.
Secondly, to be able to have a better match between dogs and potential adopters. For instance, if someone is looking for an energetic dog, they can refer to the predicted activities of the dog to find dogs that exhibit more active behaviors. This will enable prospective dog owners to make more informed.

# Background

# VGG19

In our code we used VGG19 which is a pre- trained CNN that includes 19 layers. The model was trained with more than a million images from the ImageNet database and can classify images into 1000 object categories
![image](https://github.com/hadarshloosh/DL-project/assets/129359070/bbb9dc64-8e9f-43cd-9439-5cbf737ff61c)
source: https://www.researchgate.net/figure/Illustration-of-fine-tuned-VGG19-pre-trained-CNN-model_fig1_342815128

This last fully connected layer was replaced with the layers we added to our model, and only these newly added layers were subjected to training. By doing so, we were able to leverage the pre-trained VGG model's learned features while fine-tuning the specific layers necessary for our task.

# YOLO- you only look once
Object detection algorithm that works by dividing an input image into a grid and making predictions about objects within each grid cell.
The key idea behind YOLO is to perform both object localization (finding the objects' positions) and classification (assigning labels to the objects) in a single pass through a convolutional neural network (CNN). 
This approach allows YOLO to achieve real-time object detection with impressive speed and accuracy.
* photo
  
source : https://www.datacamp.com/blog/yolo-objec 1 

We used the YOLOv8 version in which designed to be fast, accurate, and easy to use.
* photo 

# Confusion matrix
A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model.
A confusion matrix is a tabular summary of the number of correct and incorrect predictions made by a classifier. 
It can be used to evaluate the performance of a classification model through the calculation of performance metrics like accuracy, precision, recall, and F1-score.
We use the confusion matrix to see the prediction results of our model on our dataset. Eventually it helps us to see that we have imbalanced dataset and to get some conclusions on how to move on and improve the model.
* photo
  
source :  https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5
* F-measure/F1-score: The F1 score is a number between 0 and 1 and is the harmonic mean of precision and recall. F1 score sort of maintains a balance between the precision and recall for your classifier.
F1=2⋅(precision⋅recall)/(precision+recall)=(2⋅TP)/(2TP+(FP+FN))

# Gray short term (GrayST) 
A method in which we use RGB to detect movement. 
The images considered as input are three consecutive grayscale images (obtained by converting video frames from RGB to grayscale). 
By putting the three images together (like in RGB method) we can detect movement:
if we will see clear image, we will know that there wasn’t any movement. otherwise, if the image is blurry, there is probably a change between the images that can suggest that there was movement.
* photo
  
source: https://www.nature.com/articles/s41598-023-41774-2

# The dataset:
Our data set includes 662 labeled videos (few seconds each), divided to 8 different groups (resting, sitting, walking, standing (each contains 2 groups, on the floor and on the bench)). After converting the videos into images, we've got 361,502 images.
* photo
# The Process 
Our initial step involved converting our video data into images.
We utilized YOLOv3, an object detection framework, to isolate and crop the dog from each image. YOLO, which stands for "You Only Look Once," employs a deep convolutional neural network to detect objects within images. Unfortunately, both YOLOv3 and YOLOv8 couldn't detect the dog in the images (we assume that it's because our data set is dogs behind bars, which coco128 aren't familiar with).
To address this issue, we manually cropped images from our dataset that showcased dogs behind bars.
We performed a brief training session on the "roboflow" platform and subsequently integrated the newly acquired data into our project code. We continued training the YOLOv8 model using this augmented dataset to improve its ability to detect dogs in the context of being behind bars.
To enhance the accuracy of the model, we repeated the process of acquiring more data and fine-tuning the YOLOv8 model. This iterative approach aimed to provide the model with additional training examples and improve its performance in identifying dogs behind bars.
* photo

Then, we downloaded our data to our python code and trained it with augmentations: noise- up to 3% of pixels and cutout- 5 boxes with 10% size each.
Then we cropped the images using the bounding box.
* photo

Following those steps, we obtained a dataset consisting of cropped and labeled images of dogs and divided our data into: training, testing and validation. These sets were essential for training and evaluating our model effectively.
Our chosen model architecture is based on the pre-trained VGG19 network. 
After that we used under sampling and grayST techniques in order to deal with the imbalanced data and the results that we saw on the confusion matrix between standing and walking prediction.
Our last step was to make predictions for each video.

# results
During the process, we practiced and trained our model in various ways to achieve the best results and create the most suitable classification network for our project's goals.

# First practice-overfit 
After we built our net, we will see the loss function and accuracy results- at the end of the training:
* photo
  
Running our model on the test images gives us the accuracy of 96.435%.
We have identified overfitting in our model. 
We believe that this issue arises from our data splitting approach, where we divided the data into training, validation, and test sets based on individual frames.
Since each video contains numerous similar frames, there is a significant likelihood that the model trains on images that closely resemble those in the test set.

To fix the overfit, we split the data by video and not randomly by images.






**model test accuuracy after adding **gaussian noise**: 80.532**

![image](https://github.com/hadarshloosh/DL-project/assets/129359070/f4583817-f1af-44ae-9db0-f4a21fd5ab7f)

**model test accuuracy after adding **augmantation**: 80.904599%**

After few different combination, we understood that the best one is to use only the colorjitter (which make sence since randomaffine applies a combination of affine transformations to an image, including rotation, translation, shearing, and scaling. And Random perspective augmentation applies a projective transformation to an image, distorting its perspective by warping the image pixels. which is important in our data.

here is an example for one runing with all 3 augmantations:

![image](https://github.com/hadarshloosh/DL-project/assets/129359070/6545abb4-e3d2-4dfa-b974-a9b536b5980e)

# Our hand images test-set
After we trained our model to a “good enough” accuracy, we decided to try the model with our own images as a test set
We picture 3 different people with different features (nail paint, hand size, jewelry act.)
Because you can “speak” ASL with both you right and left hands, we also made one set with left hand
In the result we can see that the result are not that good, we tried to see if its “almost” correct by seeing top 3 cases of the prediction and saw that its wasn’t the case.
A good future work is to try to figure what our model was focused on while predicting the test set.

![image](https://github.com/hadarshloosh/DL-project/assets/129359070/2c2e8a6a-96a6-41c7-bce4-68fbae6dfdf9)



# Usage
1.	Download asl dataset and put in /datasets/asl
2.	Run our code in any pyton support machine. (make sure you write the right adrees to you drive)
3.	Add your hand images to the drive and copy his path to the code
4.	Try to see if the model can translate your name.
5.	See the accuracy at the result.

# Future work

if you want to use and improve our model here some ideas

1. Improve the test accuracy by changing some of the hyper parameters and more augmentation.
2. using this model as a part of a larges net which also include translation into a different language (for example it can be use in order to let people that speaks different language to be able to communicate).
3. use this model in order to translate words and sentences.
4. build a model that can "translate" from video (clip the video into picture and the use our model/ use yolo, ect..)
5. you can use this model to use it in different image prossesing and labeling as you wish. (remeber to change the num class, the data+label) 

# References:

We took our dataset from:

https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out?resource=download

The pretrain model we took from VGG19: https://www.kaggle.com/code/brussell757/american-sign-language-classification/input

The keras model we took from:
https://www.kaggle.com/code/brussell757/american-sign-language-classification

Link to our model : "https://drive.google.com/file/d/1Pkp5q2ji-ARcgkGxFlcMvIbT0nxzACyE/view?usp=sharing"
We also use many of the code and data that in the course material 


