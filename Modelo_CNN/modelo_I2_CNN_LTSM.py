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
# filename2 = 'E:/Ivan_HDD/TFG/Codigo/IvánC/modelo/prueba(LSTM_desmodel).pkl'
# filename1 = "F:/Estudiante/TFGF/IvanC/Datos_procesados/1_datos_sesion_CNN_1444px_5clases.pkl"

# dill.load_session(filename1)

# Directory where dataset is stored
# cuantos pixeles tienen las imagenes 2D (si es 1D sera px^2)
pixeles = '32' # 1D: 28 , 32 , 38 o 2D: 784, 1024 , 1444
pixeles_int = int(pixeles)
# Número de dimensiones
dir_dimension = '2Dimension'


#%%
# Libreria para Guardar y Cargar sesiones
import dill
# Ruta de los archivos
dir = "F:/Estudiante/TFGF/IvanC/Datos_train_test/"

test_labels = np.load(dir+"test_labels.npy")
test_images = np.load(dir+"test_images.npy")
train_labels = np.load(dir+"train_labels.npy")
train_images = np.load(dir+"train_images.npy")
min_samples = min(np.bincount(train_labels))

# filename_2 = 'F:/Estudiante/TFGF/IvanC/Datos_procesados/2_prueba(LSTM_desmodel).pkl'
# dill.dump_session(filename_2)

# Usar np.load para cargar las imágenes

# Model aqui va la cnn y lstm en 1d o 2d
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


# Compila el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
   

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

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2,validation_data=(test_images,test_labels))

#%%
# Guardamos sesion
# filename2 = 'E:/Ivan_HDD/TFG/Codigo/IvánC/modelo/prueba(LSTM_desmodel).pkl'
filename_2 = 'F:/Estudiante/TFGF/IvanC/Modelo_guardado/historial_2_ltsm.pkl'


with open(filename_2, 'wb') as ar:
    dill.dump(history.history, ar)

#%%
# Guardamos la red, por si la utilizazmos despues

# model.save('E:/Ivan_HDD/TFG/Codigo/IvánC/modelo/CNN_2D.h5')
# modelo = keras.models.load_model('E:/Ivan_HDD/TFG/Codigo/IvánC/modelo/CNN_2D.h5')
model.save('F:/Estudiante/TFGF/IvanC/Modelo_guardado/CNN_2D_ltsm.h5')
modelo = keras.models.load_model('F:/Estudiante/TFGF/IvanC/Modelo_guardado/CNN_2D_ltsm.h5')

# evaluation
test_loss, test_acc = modelo.evaluate(test_images,  test_labels, batch_size=batch_size, verbose=2)

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

prediction = model.predict(test_images, batch_size=batch_size, verbose=1)

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
