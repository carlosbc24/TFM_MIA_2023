#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # library used for array operations
from os import listdir # library from the system used to read files from a directory
from os import path # library used to check the veracity of files and folders
import matplotlib.pyplot as plt # library used to plot graphs and figures
from matplotlib import image # library used to import an image as a vector
import tensorflow as tf # machine learning library
from tensorflow import keras as k # keras module from tensorflow
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


# initializing variables
images = [] # list where images for training will be saved
targets = [] # list where labels for the corresponding images will be saved


# In[3]:


##### Ubicación de las imágenes de entrenamiento del datasetpara la clase 1 (HSVE)

mainpath_1 = '../Experimentos_Iteracion3/DATASETS/Dataset_Model10/HSVE_ORIGINAL_ALL_TYPES_RESCALED_256/training_set/'

print('MAIN PATH: ', mainpath_1) # to see through console path dataset


# In[4]:


##### Ubicación de las imágenes de entrenamiento del datasetpara la clase 1 (HSVE)

mainpath_0 = '../Experimentos_Iteracion3/DATASETS/Dataset_Model10/HEALTHY_BRAINS_ALL_TYPES_RESCALED_256/training_set/'

print('MAIN PATH: ', mainpath_0) # to see through console path dataset


# In[5]:


# Crear el generador de datos de imágenes
# Estas modificaciones son las que mayor rendimiento han proporcionado sobre el modelo Model11_FINAL
# Se generan imágenes sintéticas desplazando las imágenes originales hasta 9 píxeles en el eje Y y/o X y permitiendo el volteo horizontal de la imagen.
datagen = ImageDataGenerator(
    width_shift_range = [-9, 9],  # Rango de desplazamiento horizontal
    height_shift_range = [-9, 9],  # Rango de desplazamiento vertical
    horizontal_flip = True,  # Volteo horizontal aleatorio
)


# In[6]:


#population = [0, 1, 2]
#weights = [0.33, 0.33, 0.33]

# going through folders of labels for the training data
# checking the veracity of each folder to be a folder
# going through each file image within the folder
# checking the veracity of each file to be a file of interest
# importing images and labels in images and targets respectively
for folder1 in listdir(mainpath_1):
    if path.isdir(mainpath_1 + folder1):
        for folder2 in listdir(mainpath_1 + folder1):
            if path.isdir(mainpath_1 + folder1 + '/' + folder2):
                for filename in listdir(mainpath_1 + folder1 + '/' + folder2):
                    if path.isfile(mainpath_1 + folder1 + '/' + folder2 + '/' + filename) and filename != '.DS_Store' and filename != '._.DS_Store':
                        img = image.imread(mainpath_1 + folder1 + '/' + folder2 + '/' + filename)
                        images.append(img)
                        targets.append(filename[0])
                        print("IMAGEN LEÍDA: ", filename)

                        # Step 2: Here we pick the original image to perform the augmentation on
                        #image_path = 'C:/Users/carlo/Documents/Grado_Ingenieria_informatica_en_ingenieria_del_software/Curso_4/TFG/TFG_Ramon_Gomez_Recio/Jamones/TRAIN_4_TEST_1_24_DATA_AUGMENTATION/train_without_ham24/5-JCS90'
                        #image = np.expand_dims(ndimage.imread(image_path), 0)
                        img_tensor = k.preprocessing.image.img_to_array(img)
                        img_tensor = np.expand_dims(img_tensor, axis=0)

                        # step 3: pick where you want to save the augmented images
                        save_here = mainpath_1 + folder1 + '/' + folder2

                        # Step 4. we fit the original image
                        datagen.fit(img_tensor)

                        # Probability of generating a new image from the current image
                        #new_images = choices(population, weights)[0]
                        #print(new_images)

                        # step 5: iterate over images and save using the "save_to_dir" parameter
                        for x, val in zip(datagen.flow(img_tensor,                    #image we chose
                                    save_to_dir=save_here,     #this is where we figure out where to save
                                    save_prefix='_aug_'+filename,        # it will save the images as 'aug_0912' some number for every new augmented image
                                    save_format='png'),range(1)) :     # here we define a range because we want 2 augmented images otherwise it will keep looping forever I think
                            pass


# In[7]:


#population = [0, 1, 2]
#weights = [0.33, 0.33, 0.33]

# going through folders of labels for the training data
# checking the veracity of each folder to be a folder
# going through each file image within the folder
# checking the veracity of each file to be a file of interest
# importing images and labels in images and targets respectively
for folder1 in listdir(mainpath_0):
    if path.isdir(mainpath_0 + folder1):
        for folder2 in listdir(mainpath_0 + folder1):
            if path.isdir(mainpath_0 + folder1 + '/' + folder2):
                for filename in listdir(mainpath_0 + folder1 + '/' + folder2):
                    if path.isfile(mainpath_0 + folder1 + '/' + folder2 + '/' + filename) and filename != '.DS_Store' and filename != '._.DS_Store':
                        img = image.imread(mainpath_0 + folder1 + '/' + folder2 + '/' + filename)
                        images.append(img)
                        targets.append(filename[0])
                        print("IMAGEN LEÍDA: ", filename)

                        # Step 2: Here we pick the original image to perform the augmentation on
                        #image_path = 'C:/Users/carlo/Documents/Grado_Ingenieria_informatica_en_ingenieria_del_software/Curso_4/TFG/TFG_Ramon_Gomez_Recio/Jamones/TRAIN_4_TEST_1_24_DATA_AUGMENTATION/train_without_ham24/5-JCS90'
                        #image = np.expand_dims(ndimage.imread(image_path), 0)
                        img_tensor = k.preprocessing.image.img_to_array(img)
                        img_tensor = np.expand_dims(img_tensor, axis=0)

                        # step 3: pick where you want to save the augmented images
                        save_here = mainpath_0 + folder1 + '/' + folder2

                        # Step 4. we fit the original image
                        datagen.fit(img_tensor)

                        # Probability of generating a new image from the current image
                        #new_images = choices(population, weights)[0]
                        #print(new_images)

                        # step 5: iterate over images and save using the "save_to_dir" parameter
                        for x, val in zip(datagen.flow(img_tensor,                    #image we chose
                                    save_to_dir=save_here,     #this is where we figure out where to save
                                    save_prefix='_aug_'+filename,        # it will save the images as 'aug_0912' some number for every new augmented image
                                    save_format='png'),range(1)) :     # here we define a range because we want 2 augmented images otherwise it will keep looping forever I think
                            pass


# In[8]:


tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    zoom_range=0.0,
    fill_mode="nearest",
    rescale=None,
)


# In[9]:


#Loads image in from the set image path
#img = keras.preprocessing.image.load_img(image_path, target_size= (500,500))
img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Allows us to properly visualize our image by rescaling values in array
plt.figure(figsize=(3,3))
plt.imshow(img_tensor[0], cmap='gray')
plt.show()
print("Imagen de ejemplo del dataset")


# # Horizontal Flip
# (no utilizar debido a que la encefalitis herpética afecta al lóbulo frontal y temporal de forma asimétrica)

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(horizontal_flip=True)#Creates our batch of one image
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Vertical Flip
# (no tiene sentido utilizarlo dado que todas las imágenes MRI son capturadas del mismo modo)

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(vertical_flip=True)#Creates our batch of one image
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Vertical and Horizontal Flip
# (no utilizar)

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)#Creates our batch of one image
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Width Shift Range

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(width_shift_range=[-25, 25])#I also increased the number of plots
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Height Shift Range  (no utilizar)

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(height_shift_range=[-30, 30])
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Rotation Range  (no utilizar de momento)

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(rotation_range=10)
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Brightness Range (no utilizar de momento)

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(brightness_range=[0.8, 1.2])
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()
print(datagen)


# # Zoom Range  (no utilizar de momento)

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(zoom_range=[0.8, 1.0])
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Shear Range (no utilizar)

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(shear_range=50)
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Bringing It All Together Now

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(width_shift_range=[-9, 9],
                    height_shift_range=[-9, 9],
                    rotation_range=12, brightness_range=[0.8, 1.0],
                    zoom_range = [0.8, 1.0]) #Creates our batch of one image
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(10,7))#Plots our figures
for i in range(1,7):
    plt.subplot(2, 3, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # DATA AUGMENTED V2

# In[ ]:


img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=[-20, 20], 
                             rotation_range=2, brightness_range=[0.1, 1.9]) #Creates our batch of one image
pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(16, 16))#Plots our figures
for i in range(1,17):
    plt.subplot(4, 4, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_, cmap='gray')
plt.show()


# # Bringing It Together Now Transformations NEEDED!!

# In[ ]:


# Step 1. Initialize image data generator
datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=[-25, 25]) #Creates our batch of one image

# Step 2: Here we pick the original image to perform the augmentation on
#image_path = 'C:/Users/carlo/Documents/Grado_Ingenieria_informatica_en_ingenieria_del_software/Curso_4/TFG/TFG_Ramon_Gomez_Recio/Jamones/TRAIN_4_TEST_1_24_DATA_AUGMENTATION/train_without_ham24/5-JCS90'
#image = np.expand_dims(ndimage.imread(image_path), 0)
img_tensor = k.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)#Uses ImageDataGenerator to flip the images

# step 3: pick where you want to save the augmented images
save_here = 'C:/Users/carlo/Documents/Grado_Ingenieria_informatica_en_ingenieria_del_software/Curso_4/TFG/TFG_Ramon_Gomez_Recio/Jamones/TRAIN_4_TEST_1_24_DATA_AUGMENTATION'

# Step 4. we fit the original image
datagen.fit(img_tensor)

# step 5: iterate over images and save using the "save_to_dir" parameter
for x, val in zip(datagen.flow(img_tensor,                    #image we chose
        save_to_dir=save_here,     #this is where we figure out where to save
        save_prefix='aug',        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='png'),range(6)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
    pass



# In[ ]:


pic = datagen.flow(img_tensor, batch_size =1)
plt.figure(figsize=(16, 16))#Plots our figures
for i in range(1,17):
    plt.subplot(4, 4, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_)
plt.show()


# In[ ]:


from scipy import ndimage


# Step 1. Initialize image data generator
datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=[-50, 50]) #Creates our batch of one image

# Step 2: Here we pick the original image to perform the augmentation on
image_path = 'C:/Users/carlo/Documents/Grado_Ingenieria_informatica_en_ingenieria_del_software/Curso_4/TFG/TFG_Ramon_Gomez_Recio/Jamones/TRAIN_4_TEST_1_24_DATA_AUGMENTATION/train_without_ham24/5-JCS90'
image = np.expand_dims(ndimage.imread(image_path), 0)

# step 3: pick where you want to save the augmented images
save_here = 'C:/Users/carlo/Documents/Grado_Ingenieria_informatica_en_ingenieria_del_software/Curso_4/TFG/TFG_Ramon_Gomez_Recio/Jamones/TRAIN_4_TEST_1_24_DATA_AUGMENTATION'

# Step 4. we fit the original image
datagen.fit(image)

# step 5: iterate over images and save using the "save_to_dir" parameter
for x, val in zip(datagen.flow(image,                    #image we chose
        save_to_dir=save_here,     #this is where we figure out where to save
         save_prefix='aug',        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='png'),range(10)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
    pass

