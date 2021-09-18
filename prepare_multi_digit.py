
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import os
import subprocess
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Co 2 bo rieng Dataset
print(os.listdir("data/Multi_digit_data"))
train_data = pd.read_csv('data/Multi_digit_data/train.csv')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Gop 2 bo dataset

train_data1 = np.concatenate([x_train, x_test], axis=0)
labels1 = np.concatenate([y_train, y_test], axis=0)
labels2 = train_data.label
labels = np.concatenate([labels1, labels2], axis=0)
train_data2 = train_data.drop(columns='label')
images = np.concatenate([train_data1.reshape([-1,28*28]), train_data2.values], axis=0)
print(images.shape)
print(labels.shape)

digits_per_sequence = 7
number_of_sequences = 100
dataset_sequences = []
dataset_labels = []

for i in range(number_of_sequences):
    random_indices = np.random.randint(len(images), size=(digits_per_sequence,)) #Lấy random 7 cái địa chỉ ảnh
    random_digits_images = images[random_indices] #Lấy 7 ảnh từ 7 địa chỉ đó
    transformed_random_digits_images = []
    # Lật hình cho dễ nhìn
    for img in random_digits_images:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)
        transformed_random_digits_images.append(img)

    random_digits_images = np.array(transformed_random_digits_images) #ma tran anh chuoi ki tu: ahgdasjdg
    random_digits_labels = labels[random_indices] #[9 9 9 2 9 8 6]

    random_sequence = np.hstack(random_digits_images.reshape((digits_per_sequence, 28, 28))) #ảnh chuỗi kí tự đang xét
    random_labels = np.hstack(random_digits_labels.reshape(digits_per_sequence, 1)) #label của ảnh chuỗi kí tự đang xét
    
    dataset_sequences.append(random_sequence) # chuỗi các ảnh ahdjagj
    dataset_labels.append(random_labels) # chuỗi các label của ảnh 9,1,2,7,4,6,7


labels = np.array(dataset_labels)
images = np.array(dataset_sequences).reshape([-1, 28,28*digits_per_sequence,1])

#plt.figure(num='multi digit',figsize=(9,9))
#for i in range(9):
#    plt.subplot(3,3,i+1) 
#    plt.title(np.array(dataset_labels)[i])
#    plt.imshow(np.squeeze(images[i,:,:,]))
#plt.show()

for i in range (len (images)):
    label = ( "".join( str(e) for e in labels[i] ) ) # bỏ ngoặc
    images[i] = 255 - images[i]
    cv2.imwrite('data/Multi_digit_data/multi_digit_images_test/'+str(label)+'.png',images[i])



cv2.waitKey(0)




