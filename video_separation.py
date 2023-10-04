
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import shutil
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
# import kornia
# import kornia.geometry.transform as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import multiprocessing
from tqdm import tqdm
import cv2
# %matplotlib inline
import matplotlib.pyplot as plt
# from datetime import datetime
# ##checking how to convert rgb into grayscale images
# image = cv2.imread('/home/hadashadar/runs/detect/finals_crops_no_sitting/resting_n/resting_on_bench_crop/resting_on_bench0/photo0_0.jpg')
# plt.imshow(image)
#
# gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_image,cmap='gray')

############ creating a list of all the videos of resting,standing,walking
def get_deepest_folders(root_dir):
    deepest_folders = []

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            subfolders = get_deepest_folders(item_path)
            if not subfolders:
                deepest_folders.append(item_path)
            else:
                deepest_folders.extend(subfolders)

    return deepest_folders


root_path = '/home/hadashadar/runs/detect/finals_crops_no_sitting'  # Replace with the actual root folder path
deepest_folders_list = get_deepest_folders(root_path)

# print(deepest_folders_list)
# print(len(deepest_folders_list))

############### we want to use under sampling in order to work with balanced dataset
############### we will take equal numbers of videos from each class and work with the same numbers of photos in each class

# Define the number of videos you want to select from each class
num_videos_per_class = 109

# Create a dictionary to organize videos by class
videos_by_class = {}

# Initialize the dictionary with empty lists for each class
for class_name in ['walking_n', 'standing_n', 'resting_n']:
    videos_by_class[class_name] = []

# Organize the videos into the dictionary based on their class
for video_path in deepest_folders_list:
    for class_name in ['walking_n', 'standing_n', 'resting_n']:
        if class_name in video_path:
            # Split the path using '/' to extract the class name
            path_parts = video_path.split('/')
            class_index = path_parts.index(class_name)

            # Extract the subfolder and video name
            subfolder = path_parts[class_index + 1]
            video_name = path_parts[-1]

            # Construct the video path without the subfolder
            class_path = '/'.join(path_parts[:class_index + 1])

            # Add the video path to the appropriate class
            videos_by_class[class_name].append(class_path + '/' + subfolder + '/' + video_name)

# Randomly select 109 videos from each class
selected_video_paths = []
for class_name, videos in videos_by_class.items():
    if len(videos) >= num_videos_per_class:
        selected_videos = random.sample(videos, num_videos_per_class)
        selected_video_paths.extend(selected_videos)

# Now, 'selected_video_paths' contains 109 random video paths from each class
#print(selected_video_paths)
# print(len(selected_video_paths))
video_counts = {}
for class_name in ['walking_n', 'standing_n', 'resting_n']:
    count = sum(1 for video_path in selected_video_paths if class_name in video_path)
    video_counts[class_name] = count
# print(video_counts)

########### Now we want to find the minimum number of photos from all the videos files

# Initialize minimum and second minimum photo counts to large values
min_photo_count = float('inf')
second_min_photo_count = float('inf')

photo_count_list = []
# Iterate through each video folder
for video_folder in selected_video_paths:
    # Get a list of all files in the video folder
    files_in_video = os.listdir(video_folder)

    # Count the number of photo files (assuming photos have specific extensions like .jpg, .png, etc.)
    photo_count = sum(1 for file in files_in_video if file.startswith('photo') and file.endswith('.jpg'))
    photo_count_list.append(photo_count)
    # Update the minimum and second minimum photo counts
    if photo_count < min_photo_count:
        second_min_photo_count = min_photo_count
        min_photo_count = photo_count
    elif photo_count < second_min_photo_count and photo_count != min_photo_count:
        second_min_photo_count = photo_count

# The second minimum photo count across all videos
# print("Second minimum number of photos:", second_min_photo_count)
# print(photo_count_list)

################### Create a histogram

# Create the first histogram with bins of size 300
plt.figure(figsize=(10, 4))  # Adjust the figure size if needed
plt.subplot(121)
plt.hist(photo_count_list, bins=np.arange(0, max(photo_count_list) + 1, 300), color='blue', alpha=0.7, rwidth=0.5)
plt.title('Histogram of Number of Photos per Video (Bin Size: 300)')
plt.xlabel('Number of Photos')
plt.ylabel('Frequency')

# Create the second histogram with bins from 0 to 500 spaced by 50
plt.subplot(122)
plt.hist(photo_count_list, bins=np.arange(0, 501, 50), color='green', alpha=0.7, rwidth=0.5)
plt.title('Histogram of Number of Photos per Video (Bins: 0-500, Spaced by 50)')
plt.xlabel('Number of Photos')
plt.ylabel('Frequency')

# Adjust spacing between subplots
plt.tight_layout()

# Show the histograms
plt.show()


############## split the data into train test val
n_samples = len(selected_video_paths)  # The total number of samples in the dataset

## Generate a random generator with a fixed seed
rand_gen = np.random.RandomState(0)

## Generating a shuffled vector of indices
indices = np.arange(n_samples)
rand_gen.shuffle(indices)

## Split the indices into 80% train (full) / 20% test
n_samples_train_full = int(n_samples * 0.8)
train_full_indices = indices[:n_samples_train_full]
test_indices = indices[n_samples_train_full:]

## Extract the sub datasets from the full dataset using the calculated indices
train_full_set = [selected_video_paths[index] for index in train_full_indices]
test_set = [selected_video_paths[index] for index in test_indices]

## Generate a random generator with a fixed (different) seed
rand_gen = np.random.RandomState(1)

## Generating a shuffled vector of indices
indices = train_full_indices.copy()
rand_gen.shuffle(indices)

## Split the indices of the train (full) dataset into 75% train / 25% validation
n_samples_train = int(n_samples_train_full * 0.75)
train_indices = indices[:n_samples_train]
val_indices = indices[n_samples_train:n_samples_train_full]


## Extract the sub datasets from the full dataset using the calculated indices
train_set = [selected_video_paths[index] for index in train_indices]
val_set = [selected_video_paths[index] for index in val_indices]
print(len(train_set))
print(len(val_set))
print(len(test_set))

# # Set the maximum number of images per video
# max_images_per_video = 100
#
#
# video_path = '/home/hadashadar/runs/detect/finals_crops_no_sitting/walking_n/walking_on_bench_crop/walking_on_bench0'
# # Initialize a counter for the images processed in the current video path
# image_count_per_video = 0
# target_width = 224
# target_height = 224
# # Initialize a list to store sets of three grayscale images for this video
# grayscale_sets = []
#
# for root, dirs, files in os.walk(video_path):
#     for file in files:
#         if file.startswith('photo') and file.endswith('.jpg'):
#             image_path = os.path.join(root, file)
#
#             # Load the RGB image
#             rgb_image = cv2.imread(image_path)
#
#             rgb_image_resized = cv2.resize(rgb_image, (target_width, target_height))
#
#             # Convert the RGB image to grayscale
#             gray_image = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2GRAY)
#
#             # Append the grayscale image to the current set
#             grayscale_sets.append(gray_image)
#
#             # Increment the image count for the current video path
#             image_count_per_video += 1
#
#             # Check if the limit of 100 images is reached for this video path
#             if image_count_per_video >= max_images_per_video:
#                 break  # Break out of the loop for this video path
#
#     # Check if the limit of 100 images is reached for this video path
#     if image_count_per_video >= max_images_per_video:
#         break  # Break out of the loop for this video path
# # After processing all images for this video, create RGB images from sets of three grayscale images
# count = 0
# for i in range(0, len(grayscale_sets), 3):
#     # Pad the remaining grayscale images with zeros to create a complete set
#     while len(grayscale_sets) < 3:
#         grayscale_sets.append(np.zeros_like(grayscale_sets[0]))
#
#     # Create an RGB image from the three grayscale images
#     rgb_result = np.stack(grayscale_sets[i:i + 3], axis=-1)
#     plt.imshow(rgb_result)
#     plt.show()

# ############# building the class in order to organize the dataset (**after** convert 3 gray scale images into 1 rgb image)
# class DogsDataset(Dataset):
#     def __init__(self, root_path, video_list, transform=None, debug=False, our_dataset=False):
#         self.image_list = []
#         self.our_dataset = our_dataset
#
#
#         # for video_path in video_list:
#         #     for root, dirs, files in os.walk(video_path):
#         #         for file in files:
#         #             if file.startswith('photo') and file.endswith('.jpg'):
#         #                 self.image_list.append(os.path.join(root, file))
#
#         # Set the maximum number of images per video
#         max_images_per_video = 100
#         target_width = 224
#         target_height = 224
#         # Iterate through each video path
#
#         for video_path in video_list:
#             # Create a new directory for the processed images of this video
#             video_path_new = video_path.replace("finals_crops_no_sitting", "temporal_crops")  # Add '_new' to the original path
#             try:
#                 os.makedirs(video_path_new)
#             except FileExistsError:
#                 shutil.rmtree(video_path_new)
#                 os.makedirs(video_path_new)
#
#             # Initialize a counter for the images processed in the current video path
#             image_count_per_video = 0
#
#             # Initialize a list to store sets of three grayscale images for this video
#             grayscale_sets = []
#
#             for root, dirs, files in os.walk(video_path):
#                 for file in files:
#                     if file.startswith('photo') and file.endswith('.jpg'):
#                         image_path = os.path.join(root, file)
#
#                         # Load the RGB image
#                         rgb_image = cv2.imread(image_path)
#
#                         rgb_image_resized = cv2.resize(rgb_image, (target_width, target_height))
#
#                         # Convert the RGB image to grayscale
#                         gray_image = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2GRAY)
#
#                         # Append the grayscale image to the current set
#                         grayscale_sets.append(gray_image)
#
#                         # Increment the image count for the current video path
#                         image_count_per_video += 1
#
#                         # Check if the limit of 100 images is reached for this video path
#                         if image_count_per_video >= max_images_per_video:
#                             break  # Break out of the loop for this video path
#
#                 # Check if the limit of 100 images is reached for this video path
#                 if image_count_per_video >= max_images_per_video:
#                     break  # Break out of the loop for this video path
#
#             grayscale_sets = grayscale_sets[0:int(len(grayscale_sets)/3) * 3]
#
#             # After processing all images for this video, create RGB images from sets of three grayscale images
#             for i in range(0, len(grayscale_sets), 3):
#                 # Pad the remaining grayscale images with zeros to create a complete set
#                 # while len(grayscale_sets) < 3:
#                 #     grayscale_sets.append(np.zeros_like(grayscale_sets[0]))
#
#                 # Create an RGB image from the three grayscale images
#                 rgb_result = np.stack(grayscale_sets[i:i + 3], axis=-1)
#                 # Save the RGB image into the new directory for this video
#                 save_path = os.path.join(video_path_new, f"rgb_image_{i // 3}.jpg")
#                 cv2.imwrite(save_path, rgb_result)
#             for root, dirs, files in os.walk(video_path_new):
#                 for file in files:
#                     if file.startswith('rgb') and file.endswith('.jpg'):
#                         image_path = os.path.join(root, file)
#                         self.image_list.append(image_path)
#
#         self.image_list = sorted(self.image_list)
#         self.transform = transform
#         self.num_of_classes = 3
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, index):
#         # Get the image path at the given index
#         img_path = self.image_list[index]
#         parent_directory = os.path.dirname(img_path)
#         img = Image.open(img_path)  # Open the image using PIL
#         if self.transform:
#             img = self.transform(img)  # Apply any specified transformations to the image
#         # if (self.our_dataset):
#         #     return img
#         # else:
#         label = self.extract_label(parent_directory)
#         return (img, label)
#
#     def extract_label(self, filename):
#         # Extract the label from the filename
#         # Define a mapping from labels to integers
#         label_to_int_mapping = {
#             "resting": 0,
#             "standing": 1,
#             "walking": 2,
#             # Add more labels and corresponding integers as needed
#         }
#         # Convert label to int using the mapping
#         label = os.path.basename(filename).split('_')[0]
#         label_int = label_to_int_mapping.get(label, -1)  # Use -1 if label is not found in the mapping
#
#         return torch.tensor(label_int)
#
#     def get_video_name(self, index):
#         img_path = self.image_list[index]
#
#         video = os.path.dirname(img_path)
#         video = os.path.basename(video)
#
#         return video
############# building the class in order to organize the dataset (**before** convert 3 gray scale images into 1 rgb image)
class DogsDataset(Dataset):
    def __init__(self, root_path, video_list, transform=None, debug=False, our_dataset=False):
        self.image_list = []
        self.our_dataset = our_dataset


        # for video_path in video_list:
        #     for root, dirs, files in os.walk(video_path):
        #         for file in files:
        #             if file.startswith('photo') and file.endswith('.jpg'):
        #                 self.image_list.append(os.path.join(root, file))

        # Initialize a counter for the images processed in each video path
        max_images_per_video = 100  # Set the maximum number of images per video

        # Iterate through each video path
        for video_path in video_list:
            # Reset the image count for the current video path
            image_count_per_video = 0

            for root, dirs, files in os.walk(video_path):
                for file in files:
                    if file.startswith('photo') and file.endswith('.jpg'):
                        image_path = os.path.join(root, file)
                        self.image_list.append(image_path)

                        # Increment the image count for the current video path
                        image_count_per_video += 1

                        # Check if the limit of 100 images is reached for this video path
                        if image_count_per_video >= max_images_per_video:
                            break  # Break out of the loop for this video path

        # # top_dirs = os.join(root_path,)  # resting , standing , sitting , walking cropping images
        # for i in range(len(video_list)):
        #     current_top_dir = join(root_path, video_list[i])
        #     sub_dirs = os.listdir(current_top_dir)  # on_the floor , on_bench
        #     for j in range(len(sub_dirs)):
        #         current_sub_dir = join(current_top_dir, sub_dirs[j])
        #         print("5")
        #         sub_sub_dirs = os.listdir(current_sub_dir)
        #         print("6")
        #         for k in range(len(sub_sub_dirs)):  # on_the_floor_0 , .... on_bench_0
        #             print("7")
        #             current_sub_sub_dir = join(current_sub_dir, sub_sub_dirs[k])
        #             # Add image paths to the image list
        #             self.image_list.extend([join(current_sub_sub_dir, f) for f in os.listdir(current_sub_sub_dir) if
        #                            f.startswith('photo') and f.endswith('jpg')])
        self.image_list = sorted(self.image_list)
        self.transform = transform
        self.num_of_classes = 3

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # Get the image path at the given index
        img_path = self.image_list[index]
        parent_directory = os.path.dirname(img_path)
        img = Image.open(img_path)  # Open the image using PIL
        if self.transform:
            img = self.transform(img)  # Apply any specified transformations to the image
        # if (self.our_dataset):
        #     return img
        # else:
        label = self.extract_label(parent_directory)
        return (img, label)

    def extract_label(self, filename):
        # Extract the label from the filename
        # Define a mapping from labels to integers
        label_to_int_mapping = {
            "resting": 0,
            "standing": 1,
            "walking": 2,
            # Add more labels and corresponding integers as needed
        }
        # Convert label to int using the mapping
        label = os.path.basename(filename).split('_')[0]
        label_int = label_to_int_mapping.get(label, -1)  # Use -1 if label is not found in the mapping

        return torch.tensor(label_int)
    def get_video_name(self, index):
        img_path = self.image_list[index]

        video = os.path.dirname(img_path)
        video = os.path.basename(video)

        return video
# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # according to the VGG19 that is trained on the data of imagenet
])
train_dataset = DogsDataset('/home/hadashadar/runs/detect/finals_crops_no_sitting', video_list=train_set, transform=data_transform)
val_dataset = DogsDataset('/home/hadashadar/runs/detect/finals_crops_no_sitting', video_list=val_set, transform=data_transform)
test_dataset = DogsDataset('/home/hadashadar/runs/detect/finals_crops_no_sitting', video_list=test_set, transform=data_transform)

img, _ = train_dataset[7]
image = img.permute(1,2,0).numpy()
plt.imshow(image)
plt.show()

# print(len(train_dataset))
# print(len(val_dataset))
# print(len(test_dataset))


#
# from torch.utils.data import random_split
#
# # Split the dataset into training and testing sets
# train_size = int(0.8 * len(dog_dataset))  # 80% for training
# test_size = len(dog_dataset) - train_size  # Remaining 20% for testing
#
# train_dataset, test_dataset = random_split(dog_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(5)) # we want to keep on the same train,val,test samples when we used the model that we saved over again
# # Further split the training dataset into training and validation sets
# val_size = int(0.25 * len(train_dataset))  # 25% of the training set for validation
# train_size = len(train_dataset) - val_size  # Remaining 60% for training
# train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(6)) # we want to keep on the same train,val,test samples when we used the model that we saved over again
# train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size] , generator=torch.Generator().manual_seed(6)) # we want to keep on the same train,val,test samples when we used the model that we saved over again

#
weights = '/home/hadashadar/runs/detect/vgg19.pth' # the weights that we use for the vgg19 pretrained model

########## we use fine tunning and vgg19 pretrained network and change only its last layers in order to train our model
def build_model():
    model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False # we are freezing the weights of this model in order to create feature extractor
    num_features = model.classifier[6].in_features
    # we only replace the last layer of the vgg19 and adding instead our layers to train the model
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.BatchNorm1d(512),
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.BatchNorm1d(512),
        nn.Linear(512, 3),
    )

    return model

# print(f"found cuda: {torch.cuda.is_available()}\n")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #in order to set our device when working on cpu or gpu
model = build_model()
model = model.to(device)  # send the model to the device
#!pip install torchsummary
from torchsummary import summary
summary(model, input_size=(3, 224, 224))

# function to calcualte the accuracy of the model
def calculate_accuracy(model, dataloader, device, num_classes=24):
    model.eval()  # put in evaluation mode, turn of DropOut, BatchNorm uses learned statistics
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([3,3],int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # here we get the max score of the suitable label that fit to our images
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item() # testing if the prediction suits the label of the image and if it is count it and add it to the total correct
            for i,l in enumerate (labels):
                confusion_matrix[l.item(),predicted[i].item()]+=1
    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix


import torch.optim as optim
import time
# Setting the hyper parameters of the model
num_epochs = 12  # Set the number of epochs
batch_size = 16  #We tried many diffrent batch_sizes and took the best option
learning_rate = 1e-4
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss() # for the mission of classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# now we use the class that we defined in order to upload the dataset batch by batch
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#
# # Training loop
# epoch_losses_mean = {} # Dictionary to store mean losses per epoch
# epoch_length = np.arange(1, 5, 1) # Array to define the length of an epoch
# train_acc_values = []  # List to store training accuracy values
# val_acc_values = []  # List to store validation accuracy values
# counter = 0
# for epoch in range(1, num_epochs + 1):
#     model.train() # Put the model in training mode, enabling Dropout and BatchNorm to use batch statistics
#     running_loss = 0.0  # Variable to track the running loss for each epoch
#     epoch_time = time.time() # Start timing the epoch
#     # Iterate over the training dataloader
#     for i, (inputs, labels) in enumerate(train_dataloader):
#       # send them to device
#       inputs = inputs.to(device)  # Move inputs to the specified device
#       labels = labels.to(device) # Move labels to the specified device
#        # Forward pass, backward pass, and optimization steps
#       outputs = model(inputs) # Perform forward pass through the model
#       loss = criterion(outputs, labels) # calculate the loss
#       # always the same 3 steps
#       optimizer.zero_grad() # zero the parameter gradients
#       loss.backward() # backpropagation
#       optimizer.step() # update parameters
#       counter =counter +1
#       if counter % 100 == 0:
#         print(counter)
#       running_loss += loss.data.item() # Accumulate the running loss for this epoch
#
#     # Normalizing the loss by the total number of train batches
#     epoch_losses_mean[epoch] = running_loss/len(train_dataloader)
#     running_loss /= len(train_dataloader)
#     # Calculate training/test set accuracy of the existing model
#     train_accuracy, _ = calculate_accuracy(model, train_dataloader, device, 3)
#     val_accuracy, _ = calculate_accuracy(model, val_dataloader, device)
#     # Append accuracy values to the lists
#     train_acc_values.append(train_accuracy)
#     val_acc_values.append(val_accuracy)
#     # Construct the log message for the current epoch
#     log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Validation accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy, val_accuracy)
#     epoch_time = time.time() - epoch_time # Calculate the time taken for the current epoch
#     log += "Epoch Time: {:.2f} secs".format(epoch_time)  # Append the epoch time to the log message
#     print(log)   # Print the log message
#
#  # save model
#     if epoch % 1 == 0:
#         print('==> Saving model ...')
#         state = {
#             'net': model.state_dict(),
#             'epoch': epoch,
#          }
#         if not os.path.isdir('./checkpoints'):
#             os.mkdir('./checkpoints')
#         torch.save(state, f'/home/hadashadar/PycharmProjects/dog_recognition_project/checkpoints/model_batch128_before_grayst1_{epoch}')
#         print(f'saved in /home/hadashadar/PycharmProjects/dog_recognition_project/checkpoints/model_batch128_before_grayst1_{epoch}')
# print('==> Finished Training ...')



model = build_model()
model.load_state_dict(torch.load("/home/hadashadar/PycharmProjects/dog_recognition_project/checkpoints/model_batch128_before_grayst1_2")["net"]) # after we save our trained best model in the checkpint, we upload it in order to continue with the best model we have
model = model.to(device)

batch_size=16 # after we already train our model and get the best score on the valisation set, we changed the batch size on the test set in order to check the test accuracy faster
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Iterate through the DataLoader to get the shape of the data
# for batch_images, batch_labels in test_dataloader:
#     # The shape of batch_images and batch_labels will give you insights into the data
#     print(f"Batch Images Shape: {batch_images.shape}")
#     print(f"Batch Labels Shape: {batch_labels.shape}")
#     break  # Exit the loop after inspecting the first batch
# Evaluation on the test set
model.eval() # in order to evaluate the model on the test set we are in the mode of eval instead of train
total = 0
correct =0
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, predicted = torch.max(output.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy}%")


# #
# ###in order to make the nake more robast, will add gausian noise+ augmantation
# # #
# # # # adding gaussian noise to the test images and checking the accuracy
# # # a = [0.05, 0.01, 0.005] # a list of different std for the gaussian noise
# # #
# # # total=0
# # # correct=0
# # # for i in a:
# # #     with torch.no_grad():
# # #         for images, labels in test_dataloader:
# # #             noise_images = images + i* torch.randn(images.size())
# # #             images = noise_images.to(device)
# # #             labels = labels.to(device)
# # #
# # #
# # #             output = model(images)
# # #             _, predicted = torch.max(output.data, 1)
# # #
# # #             total += labels.size(0)
# # #             correct += (predicted == labels).sum().item()
# # #
# # #
# # #         accuracy = 100 * correct / total
# # #         print(f"Test Accuracy for a= {i}: {accuracy}%")
# # #
# # #
# # def max_function(x, y):
# #     if (x > y):
# #         return x
# #     else:
# #         return y
#
########### load model, calculate accuracy and confusion matrix
classes = ('0', '1', '2')
state = torch.load('/home/hadashadar/PycharmProjects/dog_recognition_project/checkpoints/model_batch128_before_grayst1_2', map_location=device)
model.load_state_dict(state['net'])
# note: `map_location` is necessary if you trained on the GPU and want to run inference on the CPU
test_accuracy, confusion_matrix = calculate_accuracy(model, test_dataloader, device)
print("test accuracy: {:.3f}%".format(test_accuracy))
confusion_matrix_percent = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:,np.newaxis])*100 ##calculate percentages
# plot confusion matrix
fig, ax = plt.subplots(1,1,figsize=(8,6))
cax = ax.matshow(confusion_matrix, cmap=plt.get_cmap('Blues'))
plt.ylabel('Actual Category')
plt.yticks(range(3), classes)
plt.xlabel('Predicted Category')
plt.xticks(range(3), classes)
plt.colorbar(cax,fraction=0.046,pad=0.04,format="%%d%%")
# Annotate each cell with counts and percentages
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, f'{confusion_matrix[i, j]}\n{confusion_matrix_percent[i, j]:.1f}%',
                 ha='center', va='center', color='black')
plt.show()
############### predicition on video
##### Evaluation on the test set
model.eval() # in order to evaluate the model on the test set we are in the mode of eval instead of train
total = 0
correct = 0
results = {}   # {"video1": {1:20, 2:30, 0:10}, "video2" : [1,2,1,1,0]}
print(len(test_dataset))
print(len(test_dataloader))
label_to_int_mapping = {
            "resting": 0,
            "standing": 1,
            "walking": 2,
            # Add more labels and corresponding integers as needed
        }
        # Convert label to int using the mapping

with torch.no_grad():
    for i in range(0, len(test_dataset)):

        image, label = test_dataset[i]
        image = image.to(device)
        label = label.to(device)
        video = test_dataset.get_video_name(i) ## using the extract video name in our class
        # Check if the video is already in the results dictionary
        if video not in results:
            results[video] = {'label': '', 0: 0, 1: 0, 2: 0}  # Initialize counts for each class and label
        output = model(image.unsqueeze(0)) ## in order to get batch_size of 1 so we can work on image per video
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.item()
        video_label = os.path.basename(video).split('_')[0] ## in order to take the label of the video
        label_int = label_to_int_mapping.get(video_label, -1)  # Use -1 if label is not found in the mapping

        # Add the label and predicted class to the results dictionary
        results[video]['label'] = label_int
        # total += labels.size(0)
        try:
            results[video][predicted] += 1
        except:
            results[video][predicted] = 1


print(results)
############ function to calculate the max - using majority vote
# for video in results:
#     counter_0 = results[video][0]
#     counter_1 = results[video][1]
#     counter_2 = results[video][2]
#
#     max1 = max_function(counter_0, counter_1)
#     max2 = max_function(max1, counter_2)
#     print(max2, "max")

########## we extract the results into a table that consists name of video , how much photos in class 0 ,class 1 and class 2 , max prediction and label of the video
import pandas as pd
# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(results).T

# Rename columns for better clarity
df.columns = ['label', '0', '1', '2']

# Add a column for video names
df.index.name = 'Video Name'

# Reset index for better formatting
df.reset_index(inplace=True)
df['Max Count'] = df[['0', '1', '2']].idxmax(axis=1)
excel_file_path = '/home/hadashadar/output_before_grayST.xlsx'
df.to_excel(excel_file_path, index=False)
print(f'DataFrame saved to {excel_file_path}')


################# we did it in a new sheet of code : need to change the path to file every time we want to access the excel sheet of the videos
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
# # Read the Excel data into a DataFrame
# df = pd.read_excel('/home/hadashadar/PycharmProjects/output.ods')
#
# # Extract the 'Predicted' and 'Label' columns
# predicted = df['Max Count']
# actual = df['label']
#
# # Calculate the confusion matrix
# confusion = confusion_matrix(actual, predicted)
# print(confusion)
# # Calculate the total number of samples
# total_samples = len(actual)
#
# # Calculate the confusion matrix with percentages
# confusion_percent = (confusion.astype('float') / confusion.sum(axis=1)[:,np.newaxis])*100 ##calculate percentages
#
# # confusion_percent = (confusion / total_samples) * 100
# print(confusion_percent)
#
# # Convert the confusion_percent matrix to a DataFrame for formatting
# confusion_percent_df = pd.DataFrame(confusion_percent, columns=range(confusion.shape[1]), index=range(confusion.shape[0]))
#
# # Add percentage formatting to the DataFrame
# confusion_percent_formatted = confusion_percent_df.applymap(lambda x: f'{x:.2f}%')
#
# # Create a heatmap of the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.set(font_scale=1.2)  # Adjust font size as needed
# sns.heatmap(confusion_percent, annot=confusion_percent_formatted, fmt='', cmap='Blues', cbar=True, square=True,
#             annot_kws={"size": 14})
#
# # Customize plot labels and title
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix for videos(Percentages)')
#
# # Show the plot
# plt.show()
#
#
