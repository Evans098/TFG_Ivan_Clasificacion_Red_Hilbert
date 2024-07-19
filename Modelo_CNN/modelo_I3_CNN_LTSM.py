import os, shutil, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, Flatten, MaxPool2D, Dropout, TimeDistributed, LSTM


import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


import pickle
import dill


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    
# Cargar el objeto desde el archivo .pkl
# filename1 = "E:/Ivan_HDD/TFG/Codigo/IvánC/datos_pulidos/prueba_soloCNN_1444px_5clases.pkl"
# filename1 = "E:/Ivan_HDD/TFG/Codigo/IvánC/Datos_2/1_datos_sesion_CNN_1444px_5clases.pkl"
# filename1 = "F:/Estudiante/TFGF/IvanC/Datos_procesados/1_datos_sesion_CNN_1444px_5clases.pkl"
# filename2 = "E:/Ivan_HDD/TFG/Codigo/IvánC/Datos_2/1_datos_sesion_CNN_1444px_5clases.pkl"
# dill.load_session(filename1)
# dill.load_session(filename1)

# Obtener las etiquetas de cada fichero
archivo_nombre_labels = 'F:/Estudiante/TFGF/IvanC/ModeloCNN_LSTM/archivos_labels.txt'
datos_labels = []
with open(archivo_nombre_labels) as f:
    for linea in f.readlines():
        datos_labels.append(linea.strip())

lista_datos = []
for i in range(len(datos_labels)):
    lista_datos.append(datos_labels[i].split())

nombre_columnas = ['fichero', 'label']

dicc_datos = []
for j in range(len(lista_datos)):
    dicc_datos.append(dict(zip(nombre_columnas,lista_datos[j])))


# Version nueva -> train test segun flujos
names_labels = [x[1] for x in lista_datos]
class_names = np.unique(names_labels)
nClasses = len(class_names)
    
# Directory where dataset is stored
# cuantos pixeles tienen las imagenes 2D (si es 1D sera px^2)
pixeles = '32' # 1D: 28 , 32 , 38 o 2D: 784, 1024 , 1444
pixeles_int = int(pixeles)
# Número de dimensiones
dir_dimension = '2Dimension'


#%%
# Ruta de los archivos
dir = "F:/Estudiante/TFGF/IvanC/Datos_train_test/datos_procesador_2/Datos_train_test/"

test_labels = np.load(dir+"test_labels.npy")
test_images = np.load(dir+"test_images.npy")
train_labels = np.load(dir+"train_labels.npy")
train_images = np.load(dir+"train_images.npy")

batch_size = 64
epochs = 50

modelo = keras.models.load_model('F:/Estudiante/TFGF/IvanC/Modelo_guardado/CNN_2D_ltsm.h5')



# evaluation
test_loss, test_acc = modelo.evaluate(test_images,  test_labels, batch_size=batch_size, verbose=2)

filename2 = 'F:/Estudiante/TFGF/IvanC/Modelo_guardado/historial_2_ltsm.pkl'

# Abrir el archivo en modo de lectura binaria ('rb')
with open(filename2, 'rb') as archivo:
    # Cargar los datos desde el archivo
    history = dill.load(archivo)

acc = history['sparse_categorical_accuracy']
val_acc = history['val_sparse_categorical_accuracy']
loss = history['loss']
val_loss = history['val_loss']

n_epochs = range(1,len(acc)+1)

plt.plot(n_epochs,loss,'bo',label='Training loss')
plt.plot(n_epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(n_epochs,acc,'bo',label='Training accuracy')
plt.plot(n_epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% 
# Predict, Clasficication Report y Confusion Matrix

prediction = modelo.predict(test_images, batch_size=batch_size, verbose=1)

from sklearn.metrics import classification_report, confusion_matrix

prediction_true_false = prediction > 0.5

prediction_1d = np.zeros(len(prediction))
for i in range(len(prediction)):
    indice = np.argmax(prediction[i])
    prediction_1d[i] = indice
    
test_labels_matrix = np.zeros((len(test_labels), nClasses))
for i in range(len(test_labels)):
    test_labels_matrix[i][test_labels[i][0]] = 1
        
con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=prediction_1d)
print('Confusion Matrix: ',con_mat)
data = con_mat

target_names = ["Class {}".format(i) for i in range(nClasses)]

classification_report = classification_report(test_labels,prediction_1d, target_names = class_names) # cambiar class_names por target_names
print('Classification Report: \n',classification_report)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


flag_con_mat = 0

if flag_con_mat == 1:
    # crear array = con_mat a mano

    df_cm = pd.DataFrame(array, class_names, class_names)
    sn.heatmap(df_cm,annot_kws={"size": 16}) # font size
    


# %%
# Medidas de rendimiento accuracy 
# througput total
# latencia media por imagen

# si flag 0 no ejecuta si flag 1 sí ejecuta
flag_run_rendimiento = 0

import time

if flag_run_rendimiento == 1:
    batch = 64
    image_list = []
    
    time1_2 = time.time()
    for i in list_test_dirs:
        im = plt.imread(i)
        image_list.append(im)
    
    image_list_np = np.asarray(image_list, dtype=np.float64)
    image_list_np_rs = image_list_np.reshape(len(list_test_dirs),pixeles_int,pixeles_int,1)
        
    predict = modelo.predict(image_list_np_rs,batch_size=batch,verbose=0)
    
    time2_2 = time.time()
    fps_2 = len(list_test_dirs)/(time2_2-time1_2)
    print('Con batch 64')
    print('FPS: ',fps_2)
    print('Tiempo total: ',time2_2-time1_2)
    
    # solo un batch para calcular latencia
    time3_2 = time.time()
    
    image_list = []
    
    for i in list_test_dirs[0:batch]:
        im = plt.imread(i)
        image_list.append(im)
    
    image_list_np = np.asarray(image_list, dtype=np.float64)
    image_list_np_rs = image_list_np.reshape(batch,pixeles_int,pixeles_int,1)
    
    time4_2 = time.time()
    
    predict = modelo.predict(image_list_np_rs,batch_size=batch,verbose=0)
    
    time5_2 = time.time()
        
    print('Latency batch: ',time5_2 - time3_2)
    print('Latency carga img batch: ',time4_2 - time3_2)
    print('Latency predict: ',time5_2 - time3_2)
else:
    print("flag_run_rendimiento = 0 No se ejecuta")
