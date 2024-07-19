"""

@author: Iván
"""
import numpy as np
# import tensorflow as tf
import os, shutil, sys
import glob
import scipy

from sklearn.model_selection import train_test_split


# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

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
# archivo_nombre_labels = 'F:/Estudiante/TFGF/IvanC/ModeloCNN_LSTM/archivos_labels.txt'

# datos_labels = []
# with open(archivo_nombre_labels) as f:
#     for linea in f.readlines():
#         datos_labels.append(linea.strip())

# lista_datos = []
# for i in range(len(datos_labels)):
#     lista_datos.append(datos_labels[i].split())

# nombre_columnas = ['fichero', 'label']

# dicc_datos = []
# for j in range(len(lista_datos)):
#     dicc_datos.append(dict(zip(nombre_columnas,lista_datos[j])))
    
# # Version nueva -> train test segun flujos
# names_labels = [x[1] for x in lista_datos]
# class_names = np.unique(names_labels)
# nClasses = len(class_names)

# # cuantos pixeles tienen las imagenes 2D (si es 1D sera px^2)
# pixeles = '32' # 1D: 28 , 32 , 38 o 2D: 784, 1024 , 1444
# pixeles_int = int(pixeles)

# # Número de dimensiones
# dir_dimension = '2Dimension'


# Libreria para Guardar y Cargar sesiones
import dill

dir_database = 'H:/Datos_TFG/8_DataBase_procesada'
# dir_databaseH = 'H:/Datos_TFG/2_DataBase_procesada'
etiquetas = ["Email", "Video", "Audio", "Chat", "FileTransfer"]
directorios = sorted_alphanumeric(os.listdir(dir_database))
# # dir_database2 = glob.glob(f"{dir_database}/*")
# # Se guarda los datos de la sessión en casa de vovler a usarlo y no tener que repetir el ejercicio anterior
# # dill.load_session(dir_database2[4]+'/datos_procesados_sesion.pkl')
num_elements = int(111460)
for label in etiquetas:
    dire = os.path.join(dir_database,label)
    # dire2 = os.path.join(dir_databaseH,label)
    # dirs_flujo = sorted_alphanumeric(os.listdir(dir_flujo))
    # data_processing(label, dire)
    if(label=="Chat"):
        d = os.path.join(dire,"Chat.npy")
        y = os.path.join(dire,"Chat_y.npy")
        chat = np.load(d)
        print(len(chat))
        chat = np.random.permutation(chat)
        chat = chat[:num_elements]
        chat = np.array((chat), dtype=np.float64)
        chat_y = np.load(y)
        chat_y = chat_y[:num_elements]
        print("Chat")
        print(len(chat))
    elif (label=="Email"):
        d = os.path.join(dire,"Email.npy")
        y = os.path.join(dire,"Email_y.npy")
        email = np.load(d)
        print(len(email))

        email = np.random.permutation(email)
        email = np.array((email), dtype=np.float64)
        email = email[:num_elements]
        email_y = np.load(y)
        email_y = email_y[:num_elements]
        # num_elements = len(email)
        # num_elements = int(num_elements)
        
        print("Email")
        print(len(email))
    elif(label == "FileTransfer"):
        num_elements2 = num_elements//4
        d = os.path.join(dire,"FileTransfer_0.npy")
        filetransferA = np.load(d)
        filetransferA = filetransferA[:num_elements]
        filetransferA = filetransferA[::2]
        filetransferA = filetransferA[::2]
        filetransfer = filetransferA
        
        d = os.path.join(dire,"FileTransfer_1.npy")
        filetransferA = np.load(d)
        filetransferA = filetransferA[:num_elements]
        filetransferA = filetransferA[1::2]
        filetransferA = filetransferA[1::2]
        filetransfer = np.concatenate((filetransfer, filetransferA))
        
        d = os.path.join(dire,"FileTransfer_2.npy")
        filetransferA = np.load(d)
        filetransferA = filetransferA[:num_elements]
        filetransferA = filetransferA[::2]
        filetransferA = filetransferA[::2]
        filetransfer = np.concatenate((filetransfer, filetransferA))
        
        d = os.path.join(dire,"FileTransfer_3.npy")
        filetransferA = np.load(d)
        filetransferA = filetransferA[:num_elements]
        filetransferA = filetransferA[1::2]
        filetransferA = filetransferA[1::2]
        filetransfer = np.concatenate((filetransfer, filetransferA))
        filetransfer = filetransfer[:num_elements]
        filetransfer = np.array((filetransfer), dtype=np.float64)
        
        y = os.path.join(dire,"FileTransfer_y.npy")
        filetransfer_y = np.load(y)
        filetransfer_y = filetransfer_y[:num_elements]
        # d2 = os.path.join(dire,"FileTransfer_p2.npy")
        # y2 = os.path.join(dire,"FileTransfer_y2.npy")
    
        # filetransfer_2 = np.load(d2)
        # filetransfer_y2 = np.load(y2)
        print("FileTransfer")
        print(len(filetransfer))
    elif(label == "Video"):
        d = os.path.join(dire,"Video.npy")
        y = os.path.join(dire,"Video_y.npy")
        video = np.load(d)
        print(len(video))
        video = np.random.permutation(video)
        video = video[:num_elements]
        video = np.array((video), dtype=np.float64)
        video_y = np.load(y)
        video_y = video_y[:num_elements]
        print("Video")
        print(len(video))
    elif(label=="Audio"):
        d = os.path.join(dire,"Audio.npy")
        y = os.path.join(dire,"Audio_y.npy")
        audio = np.load(d)
        print(len(audio))
        audio = np.random.permutation(audio)
        audio = np.array((audio), dtype=np.float64)
        audio = audio[:num_elements]
        audio_y = np.load(y)
        audio_y = audio_y[:num_elements]
        
        print("Audio")
        print(len(audio))



# # num_elements = '21400'
# # num_elements = int(num_elements)
# num_elements = len(email)
# # Generar una permutación aleatoria de los índices
# audio = np.random.permutation(audio)
# chat = np.random.permutation(chat)
# email = np.random.permutation(email)
# video = np.random.permutation(video)
# filetransfer = np.random.permutation(filetransfer)
# # filetransfer_2 = np.random.permutation(filetransfer_2)

# # Seleccionar los primeros num_elements elementos después de la permutación
# audio = audio[:num_elements]
# chat = chat[:num_elements]
# video = video[:num_elements]
# filetransfer = filetransfer[:num_elements]
# # filetransfer_2 = filetransfer_2[:num_elements]

# # Separar los elementos en pares e impares
# # filetransfer_1 = filetransfer_1[::2]  # Tomar elementos en índices pares
# # filetransfer_2 = filetransfer_2[1::2]  # Tomar elementos en índices impares


# # Etiquetas
# audio_y = audio_y[:num_elements]
# chat_y = chat_y[:num_elements]
# video_y = video_y[:num_elements]
# filetransfer_y = filetransfer_y[:num_elements]
# # filetransfer_y2 = filetransfer_y2[:num_elements]
# # Separar los elementos en pares e impares
# # filetransfer_y1 = filetransfer_y1[::2]  # Tomar elementos en índices pares
# # filetransfer_y2 = filetransfer_y2[1::2]  # Tomar elementos en índices impares


# # Se convierten en arrays datos y etiquetas
# audio = np.array((audio), dtype=np.float64)
# chat = np.array((chat), dtype=np.float64)
# email = np.array((email), dtype=np.float64)
# video = np.array((video), dtype=np.float64)
# filetransfer = np.array((filetransfer), dtype=np.float64)
# # filetransfer_2 = np.array((filetransfer_2), dtype=np.float64)




# Concatenar todos los datos
X = np.concatenate((audio, chat, email, filetransfer, video))
y = np.concatenate((audio_y, chat_y, email_y, filetransfer_y, video_y))

# Guardamos los datos de y las etiquetas
# diree = 'F:/Estudiante/TFGF/IvanC/2_Database_procesada/Datos_train_test/'
np.save(dir_database+"/Datos_train_test/"+"X2.npy", X)
np.save(dir_database+"/Datos_train_test/"+"y2.npy", y)

# Cargamos los datos y las etiquetas
# X = np.load(diree+"X.npy")
# y = np.load(diree+"y.npy")

# convertir de list de strings a array uint8, usar class_names
# for i in range(nClasses):
#     for j in range(len(y)):
#         if y[j] == class_names[i]:
#             y[j] = i

# y = np.array((y), dtype=np.uint8)

# # Si solo tiene una dimension la imagen será 784 o 1024 o 1444 si no serán 2D 28x28, 32x32 o 38x38
# if dir_dimension == '1Dimension':
#     X = X.reshape(len(X),pixeles_int,1) # 28,28,1
#     y = y.reshape(len(y),1)
# else:
#     X = X.reshape(len(X),pixeles_int,pixeles_int,1) # 28,28,1
#     y = y.reshape(len(y),1)
    

# Obetenmos los datos de train y test, tanto para los datos como para las etiquetas
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.226, random_state=42)
# np.save(dir_database+"/Datos_train_test/"+"X_train.npy", X)
# np.save(dir_database+"/Datos_train_test/"+"X_test.npy", y)
# np.save(dir_database+"/Datos_train_test/"+"y_train.npy", X)
# np.save(dir_database+"/Datos_train_test/"+"y_test.npy", y)
# X = []
# y = []
