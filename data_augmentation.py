import os
import cv2
import numpy as np
import random
from numpy.random import choice
from sklearn.utils import shuffle

def data_augmentation(images, images_path, masks_path, store_images_path='images', store_masks_path='masks'):
    isExist = os.path.exists(store_images_path)
    if not isExist:
        os.makedirs(store_images_path)
    
    isExist = os.path.exists(store_masks_path)
    if not isExist:
        os.makedirs(store_masks_path)
    
    number_array = []
    N = len(images)
    sum = 0
    n = [1,2,3,4,5]
    distribution = [0.05,0.15,0.5,0.25,0.05]
    while(1):
        number_of_merge = choice(n, p=distribution)
        if sum + number_of_merge > N:
            number_array.append(N-sum)
            break
        sum = sum + number_of_merge
        number_array.append(number_of_merge)
    index = 0
    image_num = 0
    for i in number_array:
        for j in range(i):
            if j == 0:
                result_image = cv2.imread(os.path.join(images_path, images[index]))
                result_mask = cv2.imread(os.path.join(masks_path, images[index]), cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(os.path.join(images_path, images[index]))
                mask = cv2.imread(os.path.join(masks_path, images[index]), cv2.IMREAD_GRAYSCALE)
                image, mask = random_crop_left_right(image, mask)
                result_image = concat2images(result_image, image)
                result_mask = concat2images(result_mask, mask) 
            index=index+1
        image_num = image_num + 1
        result_image, result_mask = random_drop_left_right(result_image, result_mask)
        cv2.imwrite(os.path.join(store_images_path, "image" + str(image_num) + ".jpg"), result_image)        
        cv2.imwrite(os.path.join(store_masks_path, "image" + str(image_num) + ".jpg"), result_mask)

def random_drop_left_right(image, mask):
    length = len(image[0])  
    right = length - int(random.uniform(0,1) * length *0.2)
    left = int(random.uniform(0,1) * length * 0.2)
    return image[0:,left:right], mask[0:,left:right]

def random_crop_left_right(image, mask):
    # Horizonal projection
    a = np.sum(mask, axis=0, keepdims=True)
    index = np.sort(np.argsort(a[0])[0:len(a[a == 0])])
    gaps = np.argmax(index[1:] - index[:-1])
    left = choice(index[index < index[gaps+1]])
    right = choice(index[index >= index[gaps+1]])
    new_image = image[0:, left:right]
    new_mask = mask[0:, left:right]
    return new_image, new_mask

def split_dataset(images, train_split=0.7):
    indexes = np.arange(0, len(images))
    indexes = shuffle(indexes, random_state=754)
    images = np.array(images)
    images = images[indexes]
    train_set = images[:int(len(images)*0.7)]
    val_set = images[int(len(images)*0.7):]
    return train_set, val_set

def concat2images(image1, image2):
    new_image = np.concatenate((image1,image2), axis=1)
    return new_image

if __name__ == '__main__':

    # Train and Validation path
    train_val_path = './Dataset/train/img'
    train_val_mask_path = './Dataset/train/mask'
    list_image = os.listdir(train_val_path)
    train_set, val_set = split_dataset(list_image)
    data_augmentation(list_image, train_val_path, train_val_mask_path)
