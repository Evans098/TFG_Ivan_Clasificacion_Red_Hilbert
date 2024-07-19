"""
Created on Wed Nov 11 10:02:29 2020

@author: Iván
"""
import numpy as np
import os, shutil, sys
import matplotlib.pyplot as plt
import glob

# from tensorflow.keras import layers
# from tensorflow import keras

import dill

# Función para ordenar las listas de archivos como en windows
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

audio = []
chat = []
filetransfer = []
video = []
email = []

audio_y = []
chat_y = []
filetransfer_y = []
video_y = []
email_y = []

datos_procesado=[]
datos_procesado_y=[]

def data_processing(label, dato):
    if(label=="Chat"):
        chat.append(dato)
        chat_y.append(label)
    elif (label=="Email"):
        email.append(dato)
        email_y.append(label)
    elif(label == "FileTransfer"):
        filetransfer.append(dato)
        filetransfer_y.append(label)
    elif(label == "Video"):
        video.append(dato)
        video_y.append(label)
    elif(label=="Audio"):
        audio.append(dato) 
        audio_y.append(label)


# Directorio de la base de datos sin procesar capturas_pcap_filtradas_32_H_4  /  capturas_drive_32
# base_dir = 'F:/Estudiante/TFGF/IvanC/PreProcesamiento/7_Sin_data_augmentation_con_ceros_con_hilbert/capturas_bin'
base_dir = 'E:/Ivan_HDD/TFG/Codigo/IvánC/PreProcesamiento/1_Sin_data_augmentation_con_ceros_y_sin_hilbert/capturas_bin'

# base_dir = base_dir+"/capturas_bin"
# Obtener las etiquetas de cada fichero
archivo_nombre_labels = 'F:/Estudiante/TFGF/IvanC/ModeloCNN_LSTM/archivos_labels.txt'

dir_database = 'H:/Datos_TFG/4_DataBase_procesada'

# etiquetas = ["Audio", "Chat", "Email", "Video"]
etiquetas = ["Chat", "Email"]
# etiquetas = ["FileTransfer"]
# dir_database2 = glob.glob(f"{dir_database}/*")

# Número de dimensiones
dir_dimension = '2Dimension'
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

y_list = []
# Version nueva -> train test segun flujos
names_labels = [x[1] for x in lista_datos]
class_names = np.unique(names_labels)
nClasses = len(class_names)
dirs = sorted_alphanumeric(os.listdir(base_dir))

#%%
# Proceso de generar las imágenes guardadas por flujos. Se puede saltar este paso usando np.load
list_test_dirs = []
# Para cada captura pcap se usan algunos flujos de train y otros distintos de test. El label de cada imagen viene dado por el nombre de la captura
n_f = 0
longitud = int(5462101//4)
for app in etiquetas:
    # app2 =  re.split(r'[\\/]', app)
    # app2 = app2[-1]
    print(app)
    for fichero in dirs:
        print(fichero)
        dir_flujo = os.path.join(base_dir,fichero)
        dirs_flujo = sorted_alphanumeric(os.listdir(dir_flujo))
        
        
        for k in range(len(dicc_datos)):
            if(dicc_datos[k]["fichero"] == fichero):
                label = dicc_datos[k]["label"]
                break
            else:
                label = "no_label_found"
                
        print(label)
        
        if(label==app):
            for flujo in dirs_flujo:
                dir_imagen = os.path.join(dir_flujo,flujo,dir_dimension)
                dirs_imagen = sorted_alphanumeric(os.listdir(dir_imagen))
                
                for file in dirs_imagen:
                    #print(file)
                    dir_file = os.path.join(dir_imagen,file)
                    im = plt.imread(dir_file)
                    datos_procesado.append(im)
                    datos_procesado_y.append(label)
                    # data_processing(label, im)
                    # y_list.append(label)
                    
                    
                    if(app=="FileTransfer"):
                        longitud2 = int(len(datos_procesado))
                        if(longitud<=longitud2):
                            datos_procesado=np.random.permutation(datos_procesado)
                            np.save(dir_database+"/"+app+"/"+app+"_"+str(n_f)+".npy",datos_procesado)
                            n_f = n_f + 1
                            datos_procesado=[]
                        

            print(label+" procesado correctamente")
        else:
            print("fichero distinto: "+label)
    print("hola")
    np.save(dir_database+"/"+app+"/"+app+".npy",datos_procesado)
    print("holiwiwiiwiwgdfgdgdfgfdgdgdfgdfgdfgfd")
    print(len(datos_procesado))
    datos_procesado=[]
    np.save(dir_database+"/"+app+"/"+app+"_y.npy",datos_procesado_y)
    datos_procesado_y=[]

# for fichero in dirs:
#     print(fichero)
#     dir_flujo = os.path.join(base_dir,fichero)
#     dirs_flujo = sorted_alphanumeric(os.listdir(dir_flujo))
    
    
#     for k in range(len(dicc_datos)):
#         if(dicc_datos[k]["fichero"] == fichero):
#             label = dicc_datos[k]["label"]
#             break
#         else:
#             label = "no_label_found"
            
#     print(label)
    
#     if(label=="FileTransfer"):
#         for flujo in dirs_flujo:
#             dir_imagen = os.path.join(dir_flujo,flujo,dir_dimension)
#             dirs_imagen = sorted_alphanumeric(os.listdir(dir_imagen))
            
#             for file in dirs_imagen:
#                 #print(file)
#                 dir_file = os.path.join(dir_imagen,file)
#                 im = plt.imread(dir_file)
#                 datos_procesado.append(im)
#                 datos_procesado_y.append(label)
#                 # data_processing(label, im)
#                 # y_list.append(label)
                
#                 longitud2 = int(len(datos_procesado))
                
#                 if(longitud<=longitud2):
#                     datos_procesado=np.random.permutation(datos_procesado)
#                     np.save(dir_database+"/"+label+"/"+label+"_"+str(n_f)+".npy",datos_procesado)
#                     n_f = n_f + 1
#                     datos_procesado=[]
                
#         print(label+" procesado correctamente")
#     else:
#         print("fichero distinto: "+label)
    
    
            

    
    
print("Hola")
# Guardamos sesion y datos
# Libreria para Guardar y Cargar sesiones


# Se guarda los datos de la sessión en casa de vovler a usarlo y no tener que repetir el ejercicio anterior
# dill.dump_session(dir_database2[4]+'/datos_procesados_sesion.pkl')
# np.save(dir_database2[0]+"/Audio.npy",audio)
# np.save(dir_database2[0]+"/Audio_y.npy",audio_y)
# np.save(dir_database2[1]+"/Chat.npy",chat)
# np.save(dir_database2[1]+"/Chat_y.npy",chat_y)
# np.save(dir_database2[4]+"/Email.npy",email)
# np.save(dir_database2[4]+"/Email_y.npy",email_y)
# mid_point = len(filetransfer)//2
# np.save(dir_database2[5]+"/FileTransfer1.npy",filetransfer[:mid_point])
# np.save(dir_database2[5]+"/FileTransfer_y1.npy",filetransfer_y[:mid_point])
# np.save(dir_database2[5]+"/FileTransfer2.npy",filetransfer[mid_point:])
# np.save(dir_database2[5]+"/FileTransfer_y2.npy",filetransfer_y[mid_point:])
# np.save(dir_database2[7]+"/Video.npy",video)
# np.save(dir_database2[7]+"/Video_y.npy",video_y)
# audio = np.concatenate((audio))
# Compute the mean and the variance of the training data for normalization.
print("hola")
# audio = audio.reshape(len(audio),pixeles_int,pixeles_int,1) # 28,28,1
# chat = chat.reshape(len(chat),pixeles_int,pixeles_int,1) # 28,28,1
# email = email.reshape(len(email),pixeles_int,pixeles_int,1) # 28,28,1
# filetransfer = filetransfer.reshape(len(filetransfer),pixeles_int,pixeles_int,1) # 28,28,1
# video = video.reshape(len(video),pixeles_int,pixeles_int,1) # 28,28,1
print("hola")
# X = np.concatenate((audio, chat, email, filetransfer, video))
# y = np.concatenate((audio_y, chat_y, email_y, filetransfer_y, video_y))
# y2 = np.zeros(len(y_list))
# for i in range(nClasses):
#     for j in range(len(y_list)):
#         if y_list[j] == class_names[i]:
#             y2[j] = i
# X = X.reshape(len(X),pixeles_int,pixeles_int,1) # 28,28,1
# y = y.reshape(len(y),1)
# y2 = y2.reshape(len(y2),1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.225, random_state=42)