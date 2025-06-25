#!/usr/bin/env python
# coding: utf-8

# # 1. IMPOTACIÓN DE LIBRERÍAS

# In[1]:


import os # Librería para acceder a variables del sistema
import random # Librería para generar números aleatorios
import time # Librería para contabilizar la duración del entrenamiento de cada modelo de CNN
import numpy as np # Librería empleada para operaciones de array
from os import path # Librería empleada para comprobar que existen ciertos archivos y directorios
import pandas as pd # Para gestionar datos por medio de dataframes
from os import listdir # Librería del sistema empleada para leer archivos o directorios
import seaborn as sns # Librería para realizar mapas de calor (heatmap) con matrices de confusión con sus porcentajes
import tensorflow as tf # Librería de Machine Learning
from sklearn import metrics # Para obtener métricas de los modelos de clasificación
from  matplotlib import image # Librería empleada para importar una imagen como un vector
from skimage import transform # Para modificar las dimensiones de las imágenes y así unificarlas
from keras import backend as K # backend from keras
import matplotlib.pyplot as plt # Para generar gráficos y figuras
from tensorflow import keras as k # Módulo de keras de tensorflow
from keras.utils import plot_model # Para generar un diagrama con la arquitectura del modelo de CNN construido
from keras.models import Sequential # Función de keras para inicializar un modelo secuencial
from keras.utils import to_categorical # Función de keras para obtener la matriz codificada
# mediante one-hot en el caso de la clasificación binaria
from keras.callbacks import EarlyStopping # Módulo de Early Stopping de keras
from sklearn.model_selection import train_test_split # Empleado para dividir los datos de entrenamiento en datos de entrenamiento y de validación
from tensorflow.keras.optimizers import Adam # Se importa el optimizador adam
from keras.layers import Conv2D, Flatten, Dense, Input, AveragePooling2D, Dropout, MaxPooling2D # capas para añadir
# en el modelo secuencial de keras
from tensorflow.keras.callbacks import LearningRateScheduler # Se importa el planificador del learning rate decay

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # '0' = Everything, '1' = Warnings, '2' = Errors, '3' = Fatal
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # Permitir el uso de más memoria de gpu dependiendo de la demanda de la ejecución

session = tf.compat.v1.Session(config=config) # Se establece la configuración previa


# In[ ]:


# Se muestran las unidades de CPU y GPU disponibles en el entorno actual
print("Num CPUs disponibles: ", len(tf.config.list_physical_devices('CPU')))
print("CPU disponible: ", tf.config.list_physical_devices('CPU'))

print("Num GPUs disponibles: ", len(tf.config.list_physical_devices('GPU')))
print("GPU disponible: ", tf.config.list_physical_devices('GPU'))
tf.config.experimental.list_physical_devices()


# # 2. INICIALIZACIÓN DE VARIABLES

# In[ ]:


# Se inicializan a una lista vacía las variables donde se almacenarán tanto las imágenes como las etiquetas de cada conjunto, es decir, de los conjuntos de entrenamiento, validación y test
train_images = [] # list where images for training will be saved
train_targets = [] # list where labels for the corresponding training images will be saved

val_images = [] # list where images for val will be saved
val_targets = [] # list where labels for the corresponding val images will be saved

test_images = [] # list where images for test will be saved
test_targets = [] # list where labels for the corresponding test images will be saved


# # 3. LECTURA DE IMÁGENES DEL DATASET (TRAIN-VAL-TEST)

# In[ ]:


# Ubicación de las imágenes MRI de cerebros con HSVE

class_HSVE_path = './DATASET/DATASET_Modelos1_2_3_4_5_6/HSVE_ORIGINAL_ALL_TYPES_RESCALED_256/' # Ubicación remota las imágenes MRI de la clase positiva (cerebros con la enfermedad HSVE)

print('TRAIN PATH: ', class_HSVE_path) # Se visualiza el path de esta fracción del dataset (clase1)


# In[ ]:


# Ubicación de las imágenes MRI de cerebros sanos

class_HealthyBrains_path = './DATASET/DATASET_Modelos1_2_3_4_5_6/HEALTHY_BRAINS_ALL_TYPES_RESCALED_256/' # Ubicación remota las imágenes MRI de la clase negativa (cerebros sanos)

print('TRAIN PATH: ', class_HealthyBrains_path) # Se visualiza el path de esta fracción del dataset (clase0)


# In[ ]:


# Lectura de todas las imágenes MRI de cerebros con HSVE tanto del conjunto de entrenamiento como del conjunto
# de validación y de test. Para cada imagen leida se convierten cada uno de los píxeles de la imagen al tipo
# numérico de coma flotante 'float32', se aplica una normalización estándar sobre la imagen para que
# cada una tenga una media de 0 y una desviación estándar de 1 y se asigna la etiqueta
# de la clase en cuestión. En este caso la se asigna la etiqueta 1 al tratarse de imágenes MRI de cerebros con HSVE

for folder1 in listdir(class_HSVE_path):
    if path.isdir(class_HSVE_path + folder1):
        for folder2 in listdir(class_HSVE_path + folder1):
            if path.isdir(class_HSVE_path + folder1 + '/' + folder2):
                for folder3 in listdir(class_HSVE_path + folder1 + '/' + folder2):
                    if path.isdir(class_HSVE_path + folder1 + '/' + folder2 + '/' + folder3):
                        for filename in listdir(class_HSVE_path + folder1 + '/' + folder2 + '/' + folder3):
                            print(filename)
                            if path.isfile(class_HSVE_path + folder1 + '/' + folder2 + '/' + folder3 + '/' + filename) and filename != '.DS_Store' and filename != '._.DS_Store':

                                img = image.imread(class_HSVE_path + folder1 + '/' + folder2 + '/' + folder3 + '/' + filename)

                                img = img.astype(np.float32) # Se convierte la imagen a punto flotante de 32 bits

                                imagen_normalized = (img - np.mean(img)) / np.std(img) # Se aplica una normalización estándar a cada imagen

                                if(folder1 == 'training_set'):

                                    train_images.append(imagen_normalized)
                                    train_targets.append(1)

                                    print("IMAGEN TRAIN LEÍDA: ", filename, " | LABEL: ", 1)

                                if(folder1 == 'validation_set'):

                                    val_images.append(imagen_normalized)
                                    val_targets.append(1)

                                    print("IMAGEN VALIDATION LEÍDA: ", filename, " | LABEL: ", 1)

                                if(folder1 == 'test_set'):

                                    test_images.append(imagen_normalized)
                                    test_targets.append(1)

                                    print("IMAGEN TEST LEÍDA: ", filename, " | LABEL: ", 1)

                                print(type(imagen_normalized))
                                print("SHAPE: ", imagen_normalized.shape)

num_HSVE_train = len(train_images)
num_HSVE_val = len(val_images)
num_HSVE_test = len(test_images)
print()
print("Número de imágenes de ENTRENAMIENTO LEÍDAS de la clase 1:", num_HSVE_train)
print("Número de imágenes de VALIDACIÓN LEÍDAS de la clase 1:", num_HSVE_val)
print("Número de imágenes de TEST LEÍDAS de la clase 1:", num_HSVE_test)


# In[ ]:


# Lectura de todas las imágenes MRI de cerebros SANOS tanto del conjunto de entrenamiento como del conjunto
# de validación y de test. Para cada imagen leida se convierten cada uno de los píxeles de la imagen al
# tipo numérico de coma flotante 'float32', se aplica una normalización estándar sobre la imagen para que cada
# una tenga una media de 0 y una desviación estándar de 1 y se asigna la etiqueta
# de la clase en cuestión. En este caso la se asigna la etiqueta 0 al tratarse de imágenes MRI de cerebros con HSVE

# Para que las clases sean equilibradas, como máximo se leeran tantas imágenes MRI de cerebros sanos como número de
# imágenes MRI de la enfermedad HSVE se hayan leido en la celda anterior
# Se calculan las imágenes de HSVE leidas en traning, validación y test
num_HSVE_train = len(train_images)
num_HSVE_val = len(val_images)
num_HSVE_test = len(test_images)
num_HEALTHY_train = 0
num_HEALTHY_val = 0
num_HEALTHY_test = 0

for folder1 in listdir(class_HealthyBrains_path):
    if path.isdir(class_HealthyBrains_path + folder1):
        for folder2 in listdir(class_HealthyBrains_path + folder1):
            if path.isdir(class_HealthyBrains_path + folder1 + '/' + folder2):
                for folder3 in listdir(class_HealthyBrains_path + folder1 + '/' + folder2):
                    if path.isdir(class_HealthyBrains_path + folder1 + '/' + folder2 + '/' + folder3):
                        for filename in listdir(class_HealthyBrains_path + folder1 + '/' + folder2 + '/' + folder3):
                            print(filename)
                            if path.isfile(class_HealthyBrains_path + folder1 + '/' + folder2 + '/' + folder3 + '/' + filename) and filename != '.DS_Store' and filename != '._.DS_Store':

                                img = image.imread(class_HealthyBrains_path + folder1 + '/' + folder2 + '/' + folder3 + '/' + filename)

                                img = img.astype(np.float32) # Se convierte la imagen a punto flotante de 32 bits

                                imagen_normalized = (img - np.mean(img)) / np.std(img) # Se aplica una normalización estándar a cada imagen

                                if(folder1 == 'training_set'):

                                    if(num_HEALTHY_train == num_HSVE_train):
                                        break

                                    train_images.append(imagen_normalized)
                                    train_targets.append(0)
                                    num_HEALTHY_train = num_HEALTHY_train + 1

                                    print("IMAGEN TRAIN LEÍDA: ", filename, " | LABEL: ", 0)

                                if(folder1 == 'validation_set'):

                                    if(num_HEALTHY_val == num_HSVE_val):
                                        break

                                    val_images.append(imagen_normalized)
                                    val_targets.append(0)
                                    num_HEALTHY_val = num_HEALTHY_val + 1

                                    print("IMAGEN VALIDATION LEÍDA: ", filename, " | LABEL: ", 0)

                                if(folder1 == 'test_set'):

                                    if(num_HEALTHY_test == num_HSVE_test):
                                        break

                                    test_images.append(imagen_normalized)
                                    test_targets.append(0)
                                    num_HEALTHY_test = num_HEALTHY_test + 1

                                    print("IMAGEN TEST LEÍDA: ", filename, " | LABEL: ", 0)

                                print(type(imagen_normalized))
                                print("SHAPE: ", imagen_normalized.shape)

print()
print("Número de imágenes de ENTRENAMIENTO LEÍDAS de la clase 0:", num_HEALTHY_train)
print("Número de imágenes de VALIDACIÓN LEÍDAS de la clase 0:", num_HEALTHY_val)
print("Número de imágenes de TEST LEÍDAS de la clase 0:", num_HEALTHY_test)


# # 4. PRE-PROCESADO DE LOS DATOS

# In[ ]:


# Se crean los arrays de las imágenes de entrada
X_train = np.asarray(train_images)
X_val = np.asarray(val_images)
X_test = np.asarray(test_images)

# Se crean los arrays de las etiquetas de entrada
train_targets_array = np.asarray(train_targets)
val_targets_array = np.asarray(val_targets)
test_targets_array = np.asarray(test_targets)

# # Codificación de las etiquetas para clasificación multiclase
# # y_labels_train = to_categorical(train_targets_array) # enconding the train labels with the one_hot method
# # y_labels_test = to_categorical(test_targets_array) # enconding the test labels with the one_hot method
#
# # Se divide el conjunto de entrenamiento en training y validation. Además, el parámetro stratify permite asegurar que la
# # proporción de ejemplos en cada clase sea la misma tras haberse realizado la división
# X_train, X_val, y_train, y_val = train_test_split(X_train, train_targets_array, test_size=0.21, shuffle=True, random_state=42, stratify=train_targets_array)

print("Forma de los datos de ENTRENAMIENTO: ", X_train.shape) # data.shape = (289, 256, 256, 1)
print("Forma de las etiquetas de ENTRENAMIENTO: ", train_targets_array.shape) # labels.shape = (289,)
print()

print("Forma de los datos de VALIDACIÓN: ", X_val.shape) # labels.shape = (51, 256, 256, 1)
print("Forma de las etiquetas de VALIDACIÓN: ", val_targets_array.shape) # labels.shape = (51,)
print()

print("Forma de los datos de TEST: ", X_test.shape) # data.shape = (72, 256, 256, 1)
print("Forma de las etiquetas de TEST: ", test_targets_array.shape) # labels.shape = (72,)
print()


# In[ ]:


num_clases = 2

contador_clases_training = np.bincount(train_targets_array) # Se cuenta el número de imágenes de cada clase en el conjunto de entrenamiento

# Se imprime el número de imágenes de cada clase en el conjunto de entrenamiento
for i in range(0, num_clases):
    print("Número de ejemplos de la clase",i, " en el conjunto de entrenamiento:", contador_clases_training[i])
print()

contador_clases_validation = np.bincount(val_targets_array) # Se cuenta el número de imágenes de cada clase en el conjunto de validación

# Se imprime el número de imágenes de cada clase en el conjunto de validación
for i in range(0, num_clases):
    print("Número de ejemplos de la clase",i, " en el conjunto de validación:", contador_clases_validation[i])
print()

contador_clases_test = np.bincount(test_targets_array) # Se cuenta el número de imágenes de cada clase en el conjunto de test

# Se imprime el número de imágenes de cada clase en el conjunto de test
for i in range(0, num_clases):
    print("Número de ejemplos de la clase",i, " en el conjunto de test:", contador_clases_test[i])
print()


# # 5. Previsualización de los datos de los 3 conjuntos

# In[ ]:


nombre_clases = {
    0: 'Healthy brains',
    1: 'Brains with HSVE'
}


# In[ ]:


# Previsualización del training dataset

# max_train = len(X_train)   # número máximo de imágenes del conjunto de entrenamiento
# random_index_train = random.sample(range(max_train), 6)
#
# plt.figure(figsize=(16, 12))#Plots our figures
# for index, i in enumerate(random_index_train):
#     plt.subplot(2, 3, index+1)
#     plt.imshow(X_train[i, :, :], cmap='gray')
#
#     etiqueta = train_targets_array[i]
#     nombre_etiqueta = nombre_clases[etiqueta]
#     plt.title('Etiqueta: {} - ({})'.format(etiqueta, nombre_etiqueta))
#
# plt.show()


# In[ ]:


# Previsualización del validation dataset

# max_val = len(X_val)   # número máximo de imágenes del conjunto de validación
# random_index_val = random.sample(range(max_val), 6)
#
# plt.figure(figsize=(16, 12))#Plots our figures
# for index, i in  enumerate(random_index_val):
#     plt.subplot(2, 3, index+1)
#     plt.imshow(X_val[i, :, :], cmap='gray')
#
#     etiqueta = train_targets_array[i]
#     nombre_etiqueta = nombre_clases[etiqueta]
#     plt.title('Etiqueta: {} - ({})'.format(etiqueta, nombre_etiqueta))
#
# plt.show()


# In[ ]:


# Previsualización del test dataset

# max_test = len(X_test)   # número máximo de imágenes del conjunto de test
# random_index_test = random.sample(range(max_test), 6)
#
# plt.figure(figsize=(16, 12))#Plots our figures
# for index, i in  enumerate(random_index_test):
#     plt.subplot(2, 3, index+1)
#     plt.imshow(X_test[i, :, :], cmap='gray')
#
#     etiqueta = train_targets_array[i]
#     nombre_etiqueta = nombre_clases[etiqueta]
#     plt.title('Etiqueta: {} - ({})'.format(etiqueta, nombre_etiqueta))
#
# plt.show()


# # 6. DEFINICIÓN Y CONFIGURACIÓN DEL MODELO DE CNN (Red Neuronal Convolucional) con Transfer Learning del modelo ResNet50

# In[ ]:


# Instantiate a ResNet50 model with pre-trained weights
base_model = k.applications.ResNet50(weights='imagenet', include_top=False)
base_model.layers.pop(0)

base_model.summary()

# Save the model defined previously

plot_model(base_model, to_file='arquitecturas_modelos/Model2Iteration1_baseModel_ResNet50.png', show_shapes=True)

# Freeze the base model
base_model.trainable = False

# input_layer = Input(shape=(256, 256, 1))
# x = Conv2D(3, (3, 3), padding='same', activation='relu')(input_layer)

def get_model():

    # Define the Conv2D layer
    input_layer = Conv2D(3, kernel_size = 3, activation = 'relu', input_shape=(256, 256, 1))

    # creating Keras model
    model = Sequential()

    # new input layer
    model.add(input_layer)
    # model transfer learning
    model.add(base_model)
    # Feature learning part
    # model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
    # model.add(Dropout(0.3))
    # model.add(AveragePooling2D(pool_size = (3, 3), strides = (3,3)))
    # model.add(AveragePooling2D(pool_size = (3, 3), strides = (3, 3)))
    # Classifier part
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    return model


# In[ ]:


# Se muestra un resumen de la arquitectura del modleo secuencial de keras definido en la celda anterior
with tf.device('/GPU:0'):
    model_gpu = get_model()
    # showing a summary of the layers and parameters of the model created
    model_gpu.summary()

    # Save the model defined previously
    plot_model(model_gpu, to_file='arquitecturas_modelos/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_arquitecture.png', show_shapes=True)


# # 7. ENTRENAMIENTO DEL MODELO (Training)

# In[ ]:


# Configuración de la compilación del modelo anterior:
# Se planifica el learning rate mediante la implementación 'Exponential Decay' y se emplea el optimizador 'Adam'

# lr_schedule = k.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-3,
#     decay_steps=10000,
#     decay_rate=1e-6)
# adam_optimizer = k.optimizers.Adam(learning_rate=lr_schedule)


# In[ ]:


# Se define un optimizador de tipo 'Adam' con el learning rate por defecto, es decir, con el valor 0.0001
optimizer = Adam(learning_rate=0.0001)


# In[ ]:


# Se compila el modelo mediante la GPU
with tf.device('/GPU:0'):
    model_gpu.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy','Precision'])
    print("Modelo compilado")


# In[ ]:


# Se define otro callback de 'early stopping' que permita detener el entrenamiento del modelo si el modelo no mejora el valor de su 'val_accuracy' en 5 epochs consecutivas
early_stop_val_accuracy = EarlyStopping(monitor='val_accuracy', patience=10)


# In[ ]:


# Se define otro callback de 'early stopping' que permita detener el entrenamiento del modelo si el modelo no mejora el valor de su 'val_loss' en 5 epochs consecutivas
early_stop_val_loss = EarlyStopping(monitor='val_loss', patience=10)


# In[ ]:


# # Define the learning rate scheduler function
# def lr_scheduler(epoch, lr):
#     # Set the initial learning rate
#     lr_init = 0.0001
#     # Set the decay rate
#     decay_rate = 0.1
#     # Calculate the new learning rate for this epoch
#     lr_new = lr_init / (1.0 + decay_rate * epoch)
#     # Print the new learning rate
#     print('Epoch %d: Learning rate is %f' % (epoch+1, lr_new))
#     # Return the new learning rate
#     return lr_new
#
# # Define the learning rate scheduler callback
# lr_callback = LearningRateScheduler(lr_scheduler)


# In[ ]:


# Se ENTRENA el modelo mediante la GPU
# Se especifican una serie de hiperparámetros tales como las epochs (un máximo de 50), un batch size de 8 para que
# # la GPU sea capaz de almacenar en memoria hasta 8 imágenes a la vez antes de actualizar los parámetros y se
# # especifican los 2 callbacks previamente declarados
with tf.device('/GPU:0'):

    print("Entrenando modelo de CNN...")

    start_time = time.time()

    h = model_gpu.fit(X_train, train_targets_array, epochs = 30, validation_data=(X_val, val_targets_array), batch_size = 8, callbacks=[early_stop_val_accuracy, early_stop_val_loss], verbose=1) # training method in which data and targets are passed with some specific parameters

    print("Modelo entrenado")

    end_time = time.time()
    training_time = end_time - start_time
    print("Duración del entrenamiento:", int(training_time/60), "minutos y", int(training_time%60), "segundos")


# # 8. EVALUACIÓN DEL MODELO (Test)

# In[ ]:


### Tu código para la evaluación de la red neuronal de la pregunta 2 aquí ###

# Se evalúa el modelo entrenado de red neuronal
# pack = model_gpu.evaluate(X_test, test_targets_array)
# print(pack)
# print('Pérdida (loss) obtenida en el conjunto de prueba (test):', test_loss)
# print('Exactitud (accuracy) obtenida en el conjunto de prueba (test):', test_acc)


# In[ ]:


# Proceso de evaluación
"""
getting predictions with the trained model with the test data as input and
getting the index of the max value in order to compare it with the labels
"""

# Evaluación para clasificación binaria con sigmoid

etiquetas = test_targets_array
rounded_preds = np.round(model_gpu.predict(X_test)).astype(int)
lista_unica = np.array(rounded_preds).flatten().tolist()
preds = [int(i) for i in lista_unica]
results = etiquetas == preds

print(etiquetas)
print(preds)
print(results)


# In[ ]:


# Proceso de evaluación
# """
# getting predictions with the trained model with the test data as input and
# getting the index of the max value in order to compare it with the labels
# """

# Evaluación para clasificación multiclase con softmax
# preds = np.argmax(model_gpu.predict(X_test))
# labels = np.argmax(test_targets_array) # getting the index of the max value of each column to compare it with the predictions in order to get pred results
# results = preds == labels
#
#
# print(preds)
# print(labels)
# print(results)


# In[ ]:


# Se obtienen estadísticas sencillas a partir de los resultados obtenidos en las predicciones

correct = np.sum(results == True)
incorrect = np.sum(results == False)
print("Correct: ", correct, " Correct Acc: ", (correct/len(results))*100)
print("Incorrect: ", incorrect, " Incorrect Acc: ", (incorrect/len(results))*100)


# # 9. ANÁLISIS DE LOS RESULTADOS

# In[ ]:


# Se obtiene la tabla de métricas de clasificación con la precision, F1-score, recall y accuracy

target_names = ['Cerebro sano', 'Cerebro con HSVE']
report = metrics.classification_report(etiquetas, preds, target_names=target_names, output_dict=True)
print(report)
df = pd.DataFrame(report).transpose()

# Se alm el número de imágenes de cada clase en el conjunto de entrenamiento
for i in range(0, num_clases):
    df.at["TRAINING - "+str(i), 0] = "Número de ejemplos de la clase " + str(i) + " en el conjunto de entrenamiento: " + str(contador_clases_training[i])

# Se imprime el número de imágenes de cada clase en el conjunto de validación
for i in range(0, num_clases):
    df.at["VALIDACIÓN - "+str(i), 0] = "Número de ejemplos de la clase " + str(i) + " en el conjunto de validación: " + str(contador_clases_validation[i])

# Se imprime el número de imágenes de cada clase en el conjunto de test
for i in range(0, num_clases):
    df.at["TEST - "+str(i), 0] = "Número de ejemplos de la clase " + str(i) + " en el conjunto de test: " + str(contador_clases_test[i])

df.at["Training-time", 0] = "Duración del entrenamiento: " + str(int(training_time/60)) + " minutos y " + str(int(training_time%60)) +  " segundos"

# accuracy: (tp + tn) / (p + n)
accuracy = metrics.accuracy_score(etiquetas, preds)
print("Model Accuracy: ", accuracy)
df.at["Accuracy", 0] = "Model Accuracy: " + str(accuracy)

# ROC AUC
predicted_proba = model_gpu.predict(X_test) # Se obtienen las probabilidades de predicción del modelo
auc = metrics.roc_auc_score(etiquetas, predicted_proba) # Se calcula el AUC-ROC
print('AUC-ROC:', auc)
df.at["AUC-ROC", 0] = "AUC-ROC: " + str(auc)

df.to_csv('classification_reports/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_classification_report.csv', index=True)

# Para multiclase
# predicted_proba = model_gpu.predict_proba(X_test)
# auc = metrics.roc_auc_score(etiquetas, predicted_proba, multi_class='ovr')
# print('ROC AUC: %f' % auc)


# In[ ]:


# Se visualiza la matriz de confusión
confusion_matrix = tf.math.confusion_matrix(preds, etiquetas)
cm = plt.figure(figsize=(10, 7))


x_axis_labels = ['Cerebro sano', 'Cerebro con HSVE'] # labels for x-axis [0, 1]
y_axis_labels = ['Cerebro sano', 'Cerebro con HSVE'] # labels for y-axis [0, 1]

sns.set(font_scale=2)

heat_map = sns.heatmap((confusion_matrix/np.sum(confusion_matrix, axis=1, keepdims=True)), annot=True, cmap="Blues",  xticklabels=x_axis_labels, yticklabels=y_axis_labels)
# Si no funciona, entonces probar con cmap=colormap, definido arriba

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':20}

plt.title("Matriz de Confusión", fontdict = font1)
plt.ylabel('Valores predichos', fontdict = font2)
plt.xlabel('Valores reales', fontdict = font2)

fig = heat_map.get_figure()
fig.savefig('figuras/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_confusion_matrix.png')


# In[ ]:


# Se obtienen las métricas de exactitud y pérdida obtenida en training y validation en cada epoch del entrenamiento

sns.set(font_scale=1)
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':20}

figure = pd.DataFrame(h.history).plot(figsize=(12,8))
plt.ylim(-0.05, 1.05)

plt.title("Accuracy vs loss", fontdict = font1)
plt.xlabel("Epochs", fontdict = font2)

fig = figure.get_figure()
fig.savefig('figuras/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_accuracy_vs_loss.png')


# In[ ]:


# Se obtiene la exactitud o accuracy obtenida en training y validation en cada epoch del entrenamiento
sns.set(font_scale=1)

#accuracy per epoch graph
accuracy = h.history['accuracy']
val_accuracy = h.history['val_accuracy']
epochs = h.epoch
figure = plt.figure(figsize=(6,8))
plt.title('Evolución de la exactitud (accuracy)', fontdict = font1)
plt.ylim(-0.05, 1.05)
plt.xlabel("Epochs", fontdict = font2)
plt.plot(epochs, accuracy)
plt.plot(epochs, val_accuracy)

# figure = pd.DataFrame(h.history).plot(figsize=(12,8))

fig = figure.get_figure()
fig.savefig('figuras/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_accuracy.png')


# In[ ]:


# Se obtiene la pérdida o loss obtenida en training y validation en cada epoch del entrenamiento
loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = h.epoch
figure = plt.figure(figsize=(6,8))
plt.title('Evolución de la pérdida (loss)', fontdict = font1)
plt.ylim(-0.05, 1.05)
plt.xlabel("Epochs", fontdict = font2)
plt.plot(epochs,loss)
plt.plot(epochs,val_loss)

fig = figure.get_figure()
fig.savefig('figuras/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_loss.png')


# In[ ]:


# Se visualiza la salida de las activaciones intermedias

# get_linear_filters = K.function(inputs=model_gpu.layers[0].input, outputs=model_gpu.layers[1].output)
# get_activations = K.function(inputs=model_gpu.layers[0].input, outputs=model_gpu.layers[1].output)
# filters_applied = get_linear_filters([X_train[0:2, :, :, :],0])
# activations_output = get_activations([X_train[0:2, :, :, :],0])
#
#
# filters = plt.figure(figsize=(8,8))
# for i in range(32):
#     ax = filters.add_subplot(6, 6, i + 1)
#     ax.imshow(filters_applied[0][:, :, i], cmap = 'gray')
#     plt.xticks(np.array([]))
#     plt.yticks(np.array([]))
#     plt.tight_layout()
#
# filters.savefig('20_filters_applied')
#
# activations = plt.figure(figsize=(8,8))
# for i in range(32):
#     ax = activations.add_subplot(6, 6, i + 1)
#     ax.imshow(activations_output[0][:, :, i], cmap = 'gray')
#     plt.xticks(np.array([]))
#     plt.yticks(np.array([]))
#     plt.tight_layout()

# activations.savefig('figuras/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_20_activations')


# # 10. EXPORTACIÓN DEL MODELO E HISTÓRICOS

# In[ ]:


# Se exporta el historial con la precisión y pérdida del modelo guardado
np.save('modelos_entrenados/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_HSVE_vs_Healthy_ResNet50_HISTORY.npy',h.history)


# In[ ]:


# Se guarda/exporta el modelo de CNN entrenado
model_gpu.save('modelos_entrenados/Model2Iteration1_ResNet50_TransferLearning_ALL_TYPES_DATASET_HSVE_vs_Healthy_ResNet50.h5')

