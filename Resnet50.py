### Deep Learning Project
### Roberto Daniele Cadili 01/906991
### Burak Özturan 01/944663

'''THE FOLLOWING PART WAS RUN ON OUR LOCAL MACHINES:

import numpy as np
import pandas as pd
from random import sample
from scipy.misc import imresize
import pickle, cv2
import os, sys
from sklearn.preprocessing import LabelEncoder

os.getcwd()
os.chdir("C:/Users/picch/Desktop/UNI KONSTANZ/Deep Learning Programming/Project")

def make_img_df():
    img_root = 'C:/Users/picch/Desktop/UNI KONSTANZ/Deep Learning Programming/Project/all_images/'
    img_details = pd.read_csv('C:/Users/picch/Desktop/UNI KONSTANZ/Deep Learning Programming/Project/all_data_info.csv/all_data_info.csv')
    keepers = ['Impressionism',
                'Expressionism',
                'Surrealism',
                'Cubism',
                'Abstract Art',
                'Fauvism',
                'Pop Art',
                'Art Deco',
                'Op Art',
                'Art Nouveau (Modern)']

    df_details = img_details[img_details['style'].isin(keepers)]
    img_names = df_details['new_filename'].values

    files = [f for f in os.listdir(img_root) if os.path.isfile(os.path.join(img_root, f))]

    art_list = []
    for name in files:
        if name in img_names:
            img_path = '{}{}'.format(img_root, name)
            art_list.append(img_path)

    names = []
    for path in art_list:
        img = cv2.imread(path, 1)
        try:
            img.shape
            names.append(path.lstrip(img_root))
        except AttributeError:
            continue

    styles = [df_details.loc[df_details['new_filename'] == name, 'style'].iloc[0] for name in names]

    images = ['{}{}'.format(img_root, name) for name in names]

    final_df = pd.DataFrame({'img_path':images, 'class':styles})
    final_df.to_pickle('./paths_classes_10.pkl') # path associated to class for all 32142 files of the trainset


######

def prepare_data():
    df = pd.read_pickle('./paths_classes_10.pkl')

    paths_and_classes_small, class_names = sampled_paths_classes(df)

    with open('./paths_and_classes_small.pkl', 'wb') as f: # path associated to class for 5280 files of the sampled trainset
        pickle.dump(paths_and_classes_small, f)

    class_dict = {index: art_class for index, art_class in zip(range(10), class_names)}

    with open('./class_dict.pkl', 'wb') as f: # dictionary of classes: 0:'Abstract Art', 1: 'Art Deco', etc.
        pickle.dump(class_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    images = [cv2.imread(path,1) for path, label in paths_and_classes_small]

    x = np.array([prepare_image(image) for image in images])
    y = np.array([style for path, style in paths_and_classes_small])
    
    np.savez('./images_labels_224.npz', x=x, y=y) # images' pixel matrices associated to their class
    

#######

def sampled_paths_classes(df): ## Sampling to the min class in orded to have balanced classes (528 images per class, 5280 images in tot) 
    # encode art categories as numerical values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['class'].astype('str'))
    n_classes = len(np.unique(y))
    paths_and_classes = list(zip(df['img_path'].tolist(), y))

    paths_and_classes_small = []
    for x in range(n_classes):
        temp = [(path, style) for path, style in paths_and_classes if style == x]
        samp = sample(temp, 528)
        for path, style in samp:
            paths_and_classes_small.append((path,style))

    np.random.shuffle(paths_and_classes_small)

    return paths_and_classes_small, encoder.classes_

######

def prepare_image(image, target_width=224, target_height=224, max_zoom=0.2):
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height

    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)

    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    image = image[y0:y1, x0:x1]

    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    image = imresize(image, (target_width, target_height))

    return image.astype(np.float32) / 255 


if __name__ == '__main__':
    make_img_df()
    prepare_data()
'''

##################
##################    
##################

##### THE FOLLOWING PART WAS RUN ON GOOGLE COLAB #####
    
from google.colab import drive
drive.mount('/content/gdrive')
import sys
import os
os.chdir("/content/gdrive/My Drive/project online")

import numpy as np
import pandas as pd
from random import sample
import pickle, cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
from keras.applications.resnet50 import ResNet50
from keras import applications, optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import h5py

seed = 1337
np.random.seed(seed)

input_shape = (224, 224, 3)

data = np.load('./images_labels_224.npz')
x = data['x']
y = data['y']
n_classes = len(np.unique(y))

def train_validation_split(x, y):
    # split data into training and test sets
    X_training, X_test, y_training, y_test = train_test_split(x, y, stratify=y, test_size= 0.1, random_state=1337)

    # split training into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size= 0.1, stratify=y_training, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_validation_split(x, y)

print('Train data shape: ', X_train.shape) # 90% - 4752 (4276 + 476)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape) # 10% - 528
print('Test labels shape: ', y_test.shape)

def one_hot(y_train, y_val, y_test, n_classes):
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_val = np_utils.to_categorical(y_val, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    return y_train, y_val, y_test

y_train, y_val, y_test = one_hot(y_train, y_val, y_test, n_classes)

#####

base_model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(n_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

model.summary()

filepath =  './model_final_project.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, 
                                             monitor='val_acc', 
                                             verbose=1, 
                                             save_best_only=True, 
                                             mode='max')
callbacks_list = [checkpoint]


adam = Adam(lr=0.0001)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adam,
              metrics=['accuracy'])

batch_size = 65
epochs = 50
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    verbose=1,
                    validation_data=(X_val, y_val))


model.load_weights(filepath)
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

resnet50_model = model.load_weights(filepath)

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure(figsize=(12,6))
plt.plot(history.history["acc"], '-o',label="train_acc")
plt.plot(history.history["val_acc"],'-o', label="val_acc")
plt.title("ResNet 50 model (pretrained): training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

plt.savefig("resnet50_acc_new.jpg")


plt.style.use("ggplot")
plt.figure(figsize=(12,6))
plt.plot(history.history["loss"], '-o',label="train_loss")
plt.plot(history.history["val_loss"], '-o',label="val_loss")
plt.title("ResNet 50 model (pretrained): training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="lower right")

plt.savefig("resnet50_loss_new.jpg")


y_pred=model.predict(X_test, batch_size=65, verbose=0, steps=None)

import sklearn.metrics as metrics
cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
cm_norm =   cm /cm.astype(np.float).sum(axis=1)
cm_norm =  np.around(cm_norm, decimals=2, out=None)

true_labels = ['Abstract Art', 'Art Deco','Art Nouveau','Cubism', 
                'Fauvism' , 'Expressionism', 'Impressionism' , 'Op Art'  ,
                'Pop Art' , 'Surrealism']


flip=['Abstract Art', 'Art Deco','Art Nouveau ','Cubism', 
                'Fauvism' ,     'Express.',    'Impress.' , 'Op Art'  ,
                'Pop Art' , 'Surrealism']


import seaborn as sns
import matplotlib.pyplot as plt     
fig, ax = plt.subplots(figsize=(17,15))
ax= plt.subplot()
sns.heatmap(cm_norm, annot=True, ax = ax); 
ax.set_xlabel('Predicted labels', fontsize=20);ax.set_ylabel('True labels', fontsize=20); 
ax.set_title('Confusion Matrix', fontsize=20); 
ax.xaxis.set_ticklabels(true_labels); ax.yaxis.set_ticklabels(flip);
fig.savefig('resnet_confusion_matrix.jpg')


