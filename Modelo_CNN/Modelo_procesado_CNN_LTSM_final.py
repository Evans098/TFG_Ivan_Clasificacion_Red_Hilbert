"""

@author: Iván

Referencia 1 = https://github.com/python-engineer/tensorflow-course/blob/master/05_cnn.py
Referencia 2 = Apuntes PASM de UAM EPS (Aytami)
Referencia 3 = TFM Ignacio Sotomonte
"""
import numpy as np
import tensorflow as tf
import os, shutil, sys
import matplotlib.pyplot as plt
import glob
import scipy


from tensorflow.keras import layers
from tensorflow import keras

from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, Flatten, MaxPool2D, Dropout, TimeDistributed, LSTM

import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")

# Función para ordenar las listas de archivos como en windows
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


# Directorio de la base de datos sin procesar capturas_pcap_filtradas_32_H_4  /  capturas_drive_32
# base_dir = 'F:/Estudiante/TFGF/IvanC/PreProcesamiento/capturas_pcap_filtradas_32_H_4/capturas_bin'

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

# cuantos pixeles tienen las imagenes 2D (si es 1D sera px^2)
pixeles = '32' # 1D: 28 , 32 , 38 o 2D: 784, 1024 , 1444
pixeles_int = int(pixeles)

# Número de dimensiones
dir_dimension = '2Dimension'


# Libreria para Guardar y Cargar sesiones
import dill

# Guardamos los datos de y las etiquetas
# diree = 'F:/Estudiante/TFGF/IvanC/1_Database_procesada/Datos_train_test/'
diree = 'H:/Datos_TFG/3_DataBase_procesada/Datos_train_test/'
# np.save(diree+"X2.npy", X)
# np.save(diree+"y2.npy", y)

# Cargamos los datos y las etiquetas
X = np.load(diree+"X2.npy")
y = np.load(diree+"y2.npy")

# convertir de list de strings a array uint8, usar class_names
for i in range(nClasses):
    for j in range(len(y)):
        if y[j] == class_names[i]:
            y[j] = i

y = np.array((y), dtype=np.uint8)

# Si solo tiene una dimension la imagen será 784 o 1024 o 1444 si no serán 2D 28x28, 32x32 o 38x38
if dir_dimension == '1Dimension':
    X = X.reshape(len(X),pixeles_int,1) # 28,28,1
    y = y.reshape(len(y),1)
else:
    X = X.reshape(len(X),pixeles_int,pixeles_int,1) # 28,28,1
    y = y.reshape(len(y),1)
    

# Obetenmos los datos de train y test, tanto para los datos como para las etiquetas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X = []
y = []

        
# filename1 = 'F:/Estudiante/TFGF/IvanC/DataBase_procesada/1_datos_sesion_CNN_1444px_5clases.pkl'
# # filename2 = 'F:/Estudiante/TFGF/IvanC/Datos_procesados1_datos_sesion_CNN_1444px_5clases.pkl'
# # Se guarda los datos de la sessión en casa de vovler a usarlo y no tener que repetir el ejercicio anterior
# dill.dump_session(filename1)


# Model aqui va la cnn y lstm en 1d o 2d
# Model aqui va la cnn y lstm en 1d o 2d

# from tensorflow.keras.regularizers import l2
if dir_dimension == '1Dimension':
    
    model = tensorflow.keras.models.Sequential([
    InputLayer(input_shape=(pixeles_int, 1), name='input_data'),
    Conv1D(32, 3, activation='relu'),
    MaxPool1D(pool_size=(2)),
    Conv1D(64, 3, activation='relu'),
    MaxPool1D(pool_size=(2)),
    TimeDistributed(Flatten()),
    LSTM(50,return_sequences=True),
    Dropout(0.2),
    Dense(64, activation='relu'),
    LSTM(25,return_sequences=True),
    Dropout(0.2),
    Flatten(),
    Dropout(0.2),
    Dense(5, activation='softmax', name='output_logits_tfm')
])
    
else:
    model = tensorflow.keras.models.Sequential([
    InputLayer(input_shape=(pixeles_int, pixeles_int, 1), name='input_data'),
    Conv2D(32, 3, activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    TimeDistributed(Flatten()),
    # LSTM(50,return_sequences=True),
    Dropout(0.2),
    Dense(64, activation='relu'),
    # LSTM(25,return_sequences=True),
    Dropout(0.2),
    Flatten(),
    Dropout(0.2),
    Dense(5, activation='softmax', name='output_logits_tfm')
])
   

print(model.summary())
#import sys; sys.exit()

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["sparse_categorical_accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

#%%
# Entrenamiento del modelo.

batch_size = 64
epochs = 50

# train_image y test_image tienen que ser array of float64
# train_label y test_label tienen que ser array of uint8

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2,validation_data=(X_test, y_test))

#%%
# Guardamos sesion
# filename2 = 'E:/Ivan_HDD/TFG/Codigo/IvánC/modelo/prueba(LSTM_desmodel).pkl'
# filename_2 = 'F:/Estudiante/TFGF/IvanC/Modelo_guardado/historial_2_ltsm.pkl'


# with open(filename_2, 'wb') as ar:
#     dill.dump(history.history, ar)

#%%
# Guardamos la red, por si la utilizazmos despues

# model.save('E:/Ivan_HDD/TFG/Codigo/IvánC/modelo/CNN_2D.h5')
# modelo = keras.models.load_model('E:/Ivan_HDD/TFG/Codigo/IvánC/modelo/CNN_2D.h5')
model.save('H:/Datos_TFG/3_DataBase_procesada/')
# modelo = keras.models.load_model('F:/Estudiante/TFGF/IvanC/Modelo_guardado/CNN_2D_ltsm_procesado2.h5')

# evaluation
test_loss, test_acc = model.evaluate(X_test,  y_test, batch_size=batch_size, verbose=2)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

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

prediction = model.predict(X_test, batch_size=batch_size, verbose=1)

from sklearn.metrics import classification_report, confusion_matrix

prediction_true_false = prediction > 0.5

prediction_1d = np.zeros(len(prediction))
for i in range(len(prediction)):
    indice = np.argmax(prediction[i])
    prediction_1d[i] = indice
    
test_labels_matrix = np.zeros((len(y_test), nClasses))
for i in range(len(y_test)):
    test_labels_matrix[i][y_test[i][0]] = 1
        
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=prediction_1d)
print('Confusion Matrix: ',con_mat)
data = con_mat

target_names = ["Class {}".format(i) for i in range(nClasses)]

classification_report = classification_report(y_test,prediction_1d, target_names = class_names) # cambiar class_names por target_names
print('Classification Report: \n',classification_report)


#
# 
#  
# MAtriz de confusión más visible 
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# Convertir las predicciones a etiquetas

# Crear la matriz de etiquetas verdaderas
y_test_1d = np.zeros(len(y_test))
for i in range(len(y_test)):
    y_test_1d[i] = y_test[i][0]

# Generar la matriz de confusión
con_mat = confusion_matrix(y_test_1d, prediction_1d)

# Convertir la matriz de confusión a un DataFrame para usar con seaborn
target_names = ["Class {}".format(i) for i in range(nClasses)]
con_mat_df = pd.DataFrame(con_mat, index=class_names, columns=class_names)

# Crear una figura y un eje para el gráfico
plt.figure(figsize=(10, 8))

# Crear el mapa de calor
sns.heatmap(con_mat_df, annot=True, cmap='Blues', fmt='g')

# Añadir etiquetas y título
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Mostrar el gráfico
plt.show()


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
