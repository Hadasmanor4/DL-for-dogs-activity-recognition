# DL-for-dogs-activity-recognition
**Hadar Shloosh and Hadas Manor's project about dogs' activity recognition using DL**


![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/680bc7bf-4558-458b-b575-d4f959b469b4) 
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/642aef07-1125-453c-9c7a-2e8b8afe567b)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/863e63df-877b-4990-b02f-03af2e00e1b9)

# Introduction 
Adopting dogs from animal shelters plays a crucial role in addressing multiple societal and ethical concerns.
It aids in curbing the issue of pet overpopulation by providing homes to dogs that might otherwise contribute to the strain on shelter resources or face the risk of euthanasia due to lack of space.

Despite the many successful instances of matching dogs with families, there exist instances where the compatibility between a dog and its adopted family proves challenging. These mismatches can lead to stress for both the family and the dog, potentially resulting in the dog being returned to the shelter. Data from animal shelters reveal that a significant percentage of returned dogs are due to issues related to behavior, health concerns, or simply a mismatch in lifestyle or energy levels between the dog and the adopting family.

Studies have shown that depressed dogs often experience longer stays in shelters, reduced chances of adoption, and increased susceptibility to health issues. This underscores the urgency of addressing the emotional well-being of shelter dogs through enriching activities, socialization programs, and personalized care.

# Project goals 
In our project, we aim to predict dogs' behavior through videos taken at dogs' shelter. This prediction serves two main purposes. Firstly, it provides a valuable tool for the shelter to identify dogs that may be experiencing depression or require additional attention and medical assistance. By monitoring their activity levels and behavior, we can better assess their well-being.
Secondly, to be able to have a better match between dogs and potential adopters. For instance, if someone is looking for an energetic dog, they can refer to the predicted activities of the dog to find dogs that exhibit more active behaviors. This will enable prospective dog owners to make more informed.

# Background

# VGG19

In our code we used VGG19 which is a pre- trained CNN that includes 19 layers. The model was trained with more than a million images from the ImageNet database and can classify images into 1000 object categories
![image](https://github.com/hadarshloosh/DL-project/assets/129359070/bbb9dc64-8e9f-43cd-9439-5cbf737ff61c)

source:https://www.researchgate.net/figure/Illustration-of-fine-tuned-VGG19-pre-trained-CNN-model_fig1_342815128

In this approach, we preserved the weights of all the layers in the VGG network except for the final fully connected layer. This last fully connected layer was replaced with the layers we added to our model, and only these newly added layers were subjected to training. By doing so, we were able to leverage the pre-trained VGG model's learned features while fine-tuning the specific layers necessary for our task.

# YOLO- you only look once
Object detection algorithm that works by dividing an input image into a grid and making predictions about objects within each grid cell.
The key idea behind YOLO is to perform both object localization (finding the objects' positions) and classification (assigning labels to the objects) in a single pass through a convolutional neural network (CNN). 
This approach allows YOLO to achieve real-time object detection with impressive speed and accuracy.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/b6394a1d-dbcc-48c4-a11f-0e96cca337dc)
  
We used the YOLOv8 version in which designed to be fast, accurate, and easy to use.
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/50e2433f-0598-4a18-890a-118dc4861537)

# Confusion matrix
A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model.
A confusion matrix is a tabular summary of the number of correct and incorrect predictions made by a classifier. 
It can be used to evaluate the performance of a classification model through the calculation of performance metrics like accuracy, precision, recall, and F1-score.

We use the confusion matrix to see the prediction results of our model on our dataset. Eventually it helps us to see that we have imbalanced dataset and to get some conclusions on how to move on and improve the model.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/1c673abb-952c-4d37-825e-602b5cec3ff1)

source :  https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5
* F-measure/F1-score: The F1 score is a number between 0 and 1 and is the harmonic mean of precision and recall. F1 score sort of maintains a balance between the precision and recall for your classifier.
F1=2⋅(precision⋅recall)/(precision+recall)=(2⋅TP)/(2TP+(FP+FN))

# Gray short term (GrayST) 
A method in which we use RGB to detect movement. 
The images considered as input are three consecutive grayscale images (obtained by converting video frames from RGB to grayscale). 
By putting the three images together (like in RGB method) we can detect movement.
if we will see clear image, we will know that there wasn’t any movement. otherwise, if the image is blurry, there is probably a change between the images that can suggest that there was movement.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/ff701596-2667-46f6-8f48-5cdc415a710c)
  
source: https://www.nature.com/articles/s41598-023-41774-2

# The dataset
Our data set includes 662 labeled videos (few seconds each), divided to 8 different groups (resting, sitting, walking, standing (each contains 2 groups, on the floor and on the bench)). After converting the videos into images, we've got 361,502 images.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/b42c4f8f-65ae-4f88-8de9-e22735026c56)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/b76a92eb-9e7f-4ff3-b03f-338789b869f4)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/abc92c00-6344-42c6-b1c7-f6456b29ab14)

# The Process 
Our initial step involved converting our video data into images.

We utilized YOLOv3, an object detection framework, to isolate and crop the dog from each image. YOLO, which stands for "You Only Look Once," employs a deep convolutional neural network to detect objects within images. Unfortunately, both YOLOv3 and YOLOv8 couldn't detect the dog in the images (we assume that it's because our data set is dogs behind bars, which coco128 aren't familiar with).

To address this issue, we manually cropped images from our dataset that showcased dogs behind bars.
We performed a brief training session on the "roboflow" platform and subsequently integrated the newly acquired data into our project code. We continued training the YOLOv8 model using this augmented dataset to improve its ability to detect dogs in the context of being behind bars.
To enhance the accuracy of the model, we repeated the process of acquiring more data and fine-tuning the YOLOv8 model. This iterative approach aimed to provide the model with additional training examples and improve its performance in identifying dogs behind bars.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/16408961-b760-4fc2-a946-f17dbbb47c91)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/5cb153a9-9dd4-4404-9801-58bb35844fdb)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/ff1a5d19-1034-48d8-932a-ac184fcc78cb)

Then, we downloaded our data to our python code and trained it with augmentations: noise- up to 3% of pixels and cutout- 5 boxes with 10% size each.
Then we cropped the images using the bounding box.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/5b7313f6-4d6f-4722-aaad-746bcf16ec5d)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/ac8d7a2e-38d6-4d69-800e-90feac6d8b9d)

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/c7c00763-4c8d-456c-a875-9f011cf0e327)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/5f78a88d-2b32-46cb-9ab4-31ed0afd2d5b)

Following those steps, we obtained a dataset consisting of cropped and labeled images of dogs and divided our data into: training, testing and validation. These sets were essential for training and evaluating our model effectively.
Our chosen model architecture is based on the pre-trained VGG19 network. 
After that we used under sampling and grayST techniques in order to deal with the imbalanced data and the results that we saw on the confusion matrix between standing and walking prediction.
Our last step was to make predictions for each video.

# results
During the process, we practiced and trained our model in various ways to achieve the best results and create the most suitable classification network for our project's goals.

# First practice-overfit 
After we built our net, we will see the loss function and accuracy results- at the end of the training:

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/2c2f6e27-ba5d-478b-a380-75005f580738)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/749f948c-1ade-435c-ae51-7e61d080bf49)

Running our model on the test images gives us the accuracy of 96.435%.
We have identified overfitting in our model. 
We believe that this issue arises from our data splitting approach, where we divided the data into training, validation, and test sets based on individual frames.
Since each video contains numerous similar frames, there is a significant likelihood that the model trains on images that closely resemble those in the test set.

To fix the overfit, we split the data by video and not randomly by images.

# Second practice- unbalanced data 

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/db1fa25c-fe48-4865-8b18-71460a284df6)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/ba50834c-f2a9-47ad-adf1-cfbe1c05f384)

Running our model on the test images gives us the accuracy of 94.899%.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/e874fb7d-e0ae-48e2-a66f-da0c5aa21d7d)

Here, we have valuable insights into our model and its performance. It's evident that the unbalanced nature of our data is impacting our model's training
This is apparent because there are approximately 49,000 images in class 0, no images in class 1, 21,000 in class 2, and 2,700 in class 3 for the test dataset.
As a result, the model tends to favor predictions in favor of class 0, leading to a high probability of correct predictions within our current data distribution.
To address this issue, we have made the decision to balance our data using undersampling. 

Undersampling is a technique to balance uneven datasets by keeping all the data in the minority class and decreasing the size of the majority class. It is one of several techniques data scientists can use to extract more accurate information from originally imbalanced datasets.

Additionally, given that our model lacks sufficient data to effectively recognize when a dog is in a sitting position (class 1), we have chosen not to include this class in our dataset.
To determine how many images to extract from each video (i.e., how many seconds of video to utilize) to achieve a balanced dataset, we have created a histogram of the number of images per video.
This histogram reveals that most of our videos contain fewer than 500 images each.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/5b505faf-a799-4922-aae4-fab40012823d)

# Third practice- under sampling the data and without sitting 

We chose to take 100 images for each video, for 327 videos each class with 109 videos.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/1e1f8de8-44b6-4fe4-89fd-64d95c0a9007)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/d9c079b5-6d59-40c7-8014-e787f1021829)

Running our model on the test images gives us the accuracy of 84.188%.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/8d4cb890-b0bb-4766-9831-7c415d94ae4b)

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/5035d540-8c7c-46d8-8adc-af628a38324c)

We can observe an improvement in the F1 score, but there remains some confusion between walking and standing. To address this issue, we will use grayST to detect any movement within the video frames. By capturing a clear, unblurred image, we can confidently classify it as "standing." Conversely, if we detect a blurred image, we can classify it as "walking."

For instance, when examining an image labeled as "resting":

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/1421e152-17af-44b1-ae84-862f975e4e13)

And for walking label we can see blur, unclear image due to the movement of the dog.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/53f6ac42-2089-427b-b2f1-0810df80b27f)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/fa987b37-c754-4931-acd4-ceac99d1f6aa)

# Fourth practice- GrayST

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/6091fb30-a4ab-48b6-8c6a-62bc95db6dfb)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/be1d0d8e-98e9-4e1a-a64d-57d98ccd8150)

Running our model on the test images gives us the accuracy 92.198%.
We can see that even that the input data is smaller than the original (9337 images), we can get good score for the test.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/085920b3-195d-42ee-8825-004fea607fc5)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/74543ab0-769a-4832-b299-c6912e231d12)

As we can see, there is an improvement at the score.

After training the model on our data, we now want to obtain a prediction for each video. We created a table that processes the video frame by frame, using the model to generate predictions for each image. Subsequently, we select the prediction with the highest score, which corresponds to the one with the most images that were labeled and assign that prediction to the video.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/fb8830e7-1bb2-42d3-a708-6aee82faf750)

Running our model on the test video gives us the accuracy 78.78%.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/b8829b18-d124-4c89-87dd-5dab676487f0)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/b5adb75c-d839-41ff-98fb-2b890ff37466)

For comparison, this is the confusion matrix before implementing GrayST.

![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/81410873-da03-4c7f-8e93-edd7f3c64094)
![image](https://github.com/Hadasmanor4/DL-for-dogs-activity-recognition/assets/137258791/d40eb3c0-00e1-4478-a1c3-615ad60af6fc)

As we saw during the process, we got better results (F1) after grayST then before it. 
We expected that the results of F1 after GrayST will be improved and as expected we got this result.
# Conclusion 
In our project, we aimed to predict dog behavior based on video data.
1.	Initially, we observed that using YOLO as an object detection tool on our dataset was challenging because the dogs were behind bars. Consequently, we learned that we could use a custom YOLO model for specific detection.
2.	After training the model on our dataset for a while, we realized that splitting the data into training, validation, and test sets needed to be done per video, rather than per image. This adjustment was necessary to prevent overfitting in our model.
3.	Recognizing the dataset's structure and the number of samples in each class is crucial for achieving good results with our model. Before training the model once again, we ensured that our dataset was balanced to avoid any preference for a specific class. To achieve this, we learned how to perform under-sampling.
4.	Using grayST method to deal with the confusion in prediction between two classes. 
5.	Finally predicting each video by using the majority vote of all the images within that video. This approach leads to a result of 78.78 % on the test set in the video dataset. 

# What we have learned
During our project we have learned many important skills such as:
-	About YOLO and in particular YOLOv8 and how to use it
-	YOLO customize (using roboflow)
-	Creating an efficient preprocessing class for our dataset
-	Many different writing code skills in python
-	Fine- tuning and VGG19
-	Writing a net 
-	GrayST
-	Undersampling
-	The effect of unbalanced data and ways to deal with it.
-	Confusion matrix
-	Improving our net and trying to make it more robust.

# Looking forward
- Like every net, there Is always a place for improving the accuracy.

- Label more images for the image detection to be more accurate and crop the dog correctly in more images from the data.

- Train on larger dataset with many different videos (of different dogs), in particular more data of sitting. (Our data isn’t even).

- Continue developing a system that monitoring the dogs' behavior and count each one of the dogs' activities during the day to check the dogs' natural behavior.


# References:
https://pyimagesearch.mykajabi.com 

confusion matrix: https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5

https://towardsai.net/p/l/multi-class-model-evaluation-with-confusion-matrix-and-classification-report

https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd

https://medium.com/synthesio-engineering/precision-accuracy-and-f1-score-for-multi-label-classification-34ac6bdfb404

https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62 

YOLOv8:  https://www.datacamp.com/blog/yolo-objec 

https://blog.roboflow.com/whats-new-in-yolov8/

https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/ 

https://docs.ultralytics.com/ 

VGG19:  https://www.researchgate.net/figure/Illustration-of-fine-tuned-VGG19-pre-trained-CNN-model_fig1_342815128

Augmentation: https://www.tasq.ai/glossary/augmentation/

grayST: https://www.nature.com/articles/s41598-023-41774-2
https://bmvc2022.mpi-inf.mpg.de/0355.pdf



