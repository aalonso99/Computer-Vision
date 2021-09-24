#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alejandro
"""

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################
# Importar librerías necesarias
import numpy as np
from math import ceil
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras.utils as np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import KFold

# Importar los optimizadores a usar
from keras.optimizers import RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam

# Importar el conjunto de datos
from keras.datasets import cifar100

# Importar el modelo ResNet50 y su respectiva función de preprocesamiento,
# que es necesaria pasarle a las imágenes para usar este modelo
from keras.applications.resnet import ResNet50, preprocess_input

#Establece un punto de parada
def esperar():
  input("Pulsa una tecla para continuar...")
  plt.close('all')

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A esta función solo se la llama una vez. Devuelve 4 
# vectores conteniendo, por este orden, las imágenes
# de entrenamiento, las clases de las imágenes de
# entrenamiento, las imágenes del conjunto de test y
# las clases del conjunto de test.
def cargarImagenes():
    # Cargamos Cifar100. Cada imagen tiene tamaño
    # (32 , 32, 3). Nos vamos a quedar con las
    # imágenes de 25 de las clases.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data (label_mode ='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape (train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]
    
    # Transformamos los vectores de clases en matrices.
    # Cada componente se convierte en un vector de ceros
    # con un uno en la componente correspondiente a la
    # clase a la que pertenece la imagen. Este paso es
    # necesario para la clasificación multiclase en keras.
    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)
    
    return x_train , y_train , x_test , y_test
  
#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve la accuracy de un modelo, 
# definida como el porcentaje de etiquetas bien predichas
# frente al total de etiquetas. Como parámetros es
# necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de
# keras (matrices donde cada etiqueta ocupa una fila,
# con un 1 en la posición de la clase a la que pertenece y un 0 en las demás).
def calcularAccuracy(labels, preds):
    if len(labels.shape) == 2:
      labels = np.argmax(labels, axis = 1)
      
    preds = np.argmax(preds, axis = 1)
    accuracy = sum(labels == preds)/len(labels)  
    return accuracy
  
#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución
# de la función de pérdida en el conjunto de train y
# en el de validación, y otra con la evolución de la
# accuracy en el conjunto de train y el de validación.
# Es necesario pasarle como parámetro el historial del
# entrenamiento del modelo (lo que devuelven las
# funciones fit() y fit_generator()).
def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show(block=False)
    
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.show(block=False)
    
#########################################################################
################## FUNCIÓN PARA LEER LAS IMÁGENES #######################
#########################################################################

# Dado un fichero train.txt o test.txt y el path donde se encuentran los
# ficheros y las imágenes, esta función lee las imágenes
# especificadas en ese fichero y devuelve las imágenes en un vector y 
# sus clases en otro.

def leerImagenes(vec_imagenes, path):
  clases = np.array([img.split('/')[0] for img in vec_imagenes])
  imagenes = np.array([img_to_array(load_img(path + "/" + img, 
                                             target_size = (224, 224))) 
                       for img in vec_imagenes])
  return imagenes, clases
    
#########################################################################
############# FUNCIÓN PARA CARGAR EL CONJUNTO DE DATOS ##################
#########################################################################

# Usando la función anterior, y dado el path donde se encuentran las
# imágenes y los archivos "train.txt" y "test.txt", devuelve las 
# imágenes y las clases de train y test para usarlas con keras
# directamente.

def cargarDatos(path):
  # Cargamos los ficheros
  train_images = np.loadtxt(path + "/train.txt", dtype = str)
  test_images = np.loadtxt(path + "/test.txt", dtype = str)
  
  # Leemos las imágenes con la función anterior
  train, train_clases = leerImagenes(train_images, path)
  test, test_clases = leerImagenes(test_images, path)
  
  # Pasamos los vectores de las clases a matrices 
  # Para ello, primero pasamos las clases a números enteros
  clases_posibles = np.unique(np.copy(train_clases))
  for i in range(len(clases_posibles)):
    train_clases[train_clases == clases_posibles[i]] = i
    test_clases[test_clases == clases_posibles[i]] = i

  # Después, usamos la función to_categorical()
  train_clases = np_utils.to_categorical(train_clases, 200)
  test_clases = np_utils.to_categorical(test_clases, 200)
  
  # Barajar los datos
  train_perm = np.random.permutation(len(train))
  train = train[train_perm]
  train_clases = train_clases[train_perm]

  test_perm = np.random.permutation(len(test))
  test = test[test_perm]
  test_clases = test_clases[test_perm]
  
  return train, train_clases, test, test_clases


def normalizar_matriz(im):
  minimo = min(im.ravel())
  maximo = max(im.ravel())
  rango = maximo-minimo
  return (im-minimo)/rango

def pintaI(im, figsize=(5,5)):
  im_norm = normalizar_matriz(im)
  cmap = 'gray' if len(im.shape) == 2 else None
  plt.figure(figsize=figsize)
  plt.axis('off')
  plt.imshow( im_norm, cmap=cmap, vmin=0.0, vmax=1.0 )
  plt.show(block=False)

def pintaMIE(vim, titulos, ncol = 2, figsize=(12,12)):
  fig = plt.figure(figsize=figsize)
  #fig = plt.figure()
  for k, im in enumerate(vim):
    sub = fig.add_subplot( ceil(len(vim)/ncol), ncol, k + 1 )
    plt.axis('off')
    plt.title(titulos[k])
    cmap = 'gray' if len(im.shape) == 2 else None
    sub.imshow(normalizar_matriz(im), cmap=cmap)
  plt.show(block=False)


####Función que entrena un modelo a partir de un conjunto de datos aumentado.
#### Permite especificar los parámetros del entrenamiento y si desea que se muestre
#### la evolución por época en pantalla
def fit_augmented_data(model, x_train, y_train, batch_size, epochs, callbacks=[], verbose=1, show_plots=True):
  x_train_red, x_validation, y_train_red, y_validation = train_test_split(x_train, y_train, test_size=0.1)
  datagen = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True,
      zoom_range=0.2,
      horizontal_flip=True,
      )
  datagen_validation = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True
      )
  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  datagen.fit(x_train)
  datagen_validation.fit(x_train)
  hist = model.fit(
            datagen.flow(x_train_red, y_train_red, batch_size=batch_size),
            validation_data=datagen_validation.flow(x_validation, y_validation, batch_size=batch_size),
            steps_per_epoch=len(x_train)*0.9/batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_steps=len(x_train)*0.1/batch_size,
            verbose=verbose
            )
  
  if show_plots:
    mostrarEvolucion(hist)

  return model

#### Prueba la precisión de predicción de un modelo.
#### Las imágenes de test se normalizan a partir de la media y desviación
#### típica del conjunto de entrenamiento
def test_model(model, x_train, x_test, y_test, verbose=1):
  ###Evaluación en el conjunto test normalizado
  datagen_test = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True,
      )
  datagen_test.fit(x_train)
  y_pred = model.predict(
                  datagen_test.flow(x_test, batch_size=1, shuffle=False),
                  steps=len(x_test)
                )
  acc = calcularAccuracy(y_test, y_pred)

  if verbose == 1:
    print('Test accuracy:', acc)

  return acc

class P2():
  
  def init(self):
    pass
    
  """
  Funciones para el apartado 1
  """
  ###Carga de las imágenes de CIFAR100
  def cargaCIFAR100(self):
    self.x_train , self.y_train , self.x_test , self.y_test = cargarImagenes()
    
    #Barajado de las imágenes
    from random import shuffle
    ind_train = list(range(len(self.x_train)))
    ind_test = list(range(len(self.x_test)))
    shuffle(ind_train)
    shuffle(ind_test)
    
    self.x_train = self.x_train[ind_train]
    self.y_train = self.y_train[ind_train]
    self.x_test = self.x_test[ind_test]
    self.y_test = self.y_test[ind_test]
    
  def pruebaVisualizar(self):
    ###Muestra algunas imágenes de entrenamiento
    fig = plt.figure(figsize=(15,25))
    for i in range(10):
      fig.add_subplot(5, 2, i+1)
      plt.title(self.y_train[i])
      plt.imshow(self.x_train[i])
      plt.show(block=False)
    ###Muestra algunas imágenes de test  
    fig = plt.figure(figsize=(15,25))
    for i in range(10):
      fig.add_subplot(5, 2, i+1)
      plt.title(self.y_test[i])
      plt.imshow(self.x_test[i])
      plt.show(block=False)
  
  def BaseNet(self):
    # Define Sequential model with 3 layers
    model = keras.Sequential()
    #De acuerdo con la documentación de Keras, los pesos se inicializan con
    # distribución Glorot Uniforme por defecto (https://keras.io/api/layers/convolution_layers/convolution2d/)
    model.add(layers.Conv2D(6, kernel_size=5, activation='relu',
                            input_shape=(32, 32, 3), name='Conv2d_1'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='Max_Pooling_1'))
    model.add(layers.Conv2D(16, kernel_size=(5, 5),
                    activation='relu', name='Conv2d_2'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='Max_Pooling_2'))
    model.add(layers.Flatten(name='Flatten'))
    model.add(layers.Dense(50, activation='relu', name='Dense_1'))
    model.add(layers.Dense(25, activation='softmax', name='Dense_2'))
    return model
  
  #Devuelve la precisión media al aplicar validación cruzada para entrenar un modelo
  def cross_val_optimizers(self, model, x_train, y_train, batch_size, epochs, n_splits=5, verbose=0, show_plots=True):
    w = model.get_weights()
    kf = KFold(n_splits=n_splits)
    sum_accuracy = 0
    for train_index, test_index in kf.split(x_train):
      model.set_weights(w)
      x_cross_train, x_cross_test = x_train[train_index], x_train[test_index]
      y_cross_train, y_cross_test = y_train[train_index], y_train[test_index]
  
      hist = model.fit(x_cross_train, y_cross_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_split=0.1,
              verbose=verbose
              )
      if show_plots:
        mostrarEvolucion(hist)
  
      sum_accuracy += test_model(model, x_cross_train, x_cross_test, y_cross_test)
    mean_accuracy = sum_accuracy/n_splits
    return mean_accuracy
  
  #Devuelve el optimizador con la mayor precisión media obtenida en la validación
  # cruzada y una lista con la precisión media de cada optimizador
  def test_optimizers(self, model, optimizers, x_train, y_train, batch_size, epochs, n_splits=5, verbose=0, show_plots=True):
    weights = model.get_weights()   #Guardamos los pesos para poder reiniciarlos en cada iteración
    mean_accuracies = []            #Almacenará las precisiones medias de cada optimizador
    best_mean_accuracy = -1
    best_optimizer = None
    for i, opt in enumerate(optimizers):
      print("Cross-validating optimizer", i+1)
      model.set_weights(weights)     #Reiniciamos los pesos
      model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])
      
      mean_accuracy = self.cross_val_optimizers(model, x_train, y_train, batch_size, epochs, n_splits=n_splits, 
                                           verbose=verbose, show_plots=show_plots)
      mean_accuracies.append(mean_accuracy)
  
      if mean_accuracy > best_mean_accuracy:
        best_mean_accuracy = mean_accuracy
        best_optimizer = opt
  
    return best_optimizer, mean_accuracies
  
  ### Comparación de distintos optimizadores con validación cruzada
  def cross_comp_opt(self):
    optimizers = [
                  RMSprop(), 
                  Adam(),
                  Adadelta(),
                  Adagrad(),
                  Adamax(),
                  Nadam()  
    ]
    batch_size = 64
    epochs = 30
    # Imprime el optimizador con mejor media y un array con la medias obtenidas con cada optimizador
    print(self.test_optimizers(self.BaseNet(), optimizers, self.x_train, self.y_train, batch_size, epochs))

  ###Normalización, aumento de datos y prueba
  def train_test_basenet(self):
    batch_size = 64
    epochs = 30
    model = self.BaseNet()
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.RMSprop(),
                    metrics=['accuracy'])
    model = fit_augmented_data(model, self.x_train, self.y_train, batch_size, epochs,
                               callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)])

    print("Test accuracy:", test_model(model, self.x_train, self.x_test, self.y_test, verbose=0))
    
  def apartado1(self):
    self.cargaCIFAR100()
    self.pruebaVisualizar()
    esperar()
    self.cross_comp_opt()
    esperar()
    self.train_test_basenet()
    esperar()
    
  """
  Funciones para el apartado 2
  """
  def BaseNet2_1(self):
    model = keras.Sequential()
    
    model.add(layers.Conv2D(32, kernel_size=3, input_shape=(32, 32, 3), name='Conv2D_1'))
    model.add(layers.Conv2D(32, kernel_size=3, name='Conv2D_2'))
    model.add(layers.BatchNormalization(momentum=0.99, name='BatchNorm_1'))
    model.add(layers.Activation(keras.activations.relu, name='Activation_1'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_1'))
  
    model.add(layers.Conv2D(64, kernel_size=3, name='Conv2D_3'))
    model.add(layers.BatchNormalization(momentum=0.99, name='BatchNorm_2'))
    model.add(layers.Activation(keras.activations.relu, name='Activation_2'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_2'))
  
    model.add(layers.Flatten(name='Flatten'))
    model.add(layers.Dropout(0.4, name='Dropout_0_4'))
    model.add(layers.Dense(25, activation='softmax', name='DenseSoftmax'))
    return model 

  def train_test_basenet2_1(self):
    model = self.BaseNet2_1()
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=self.optimizer,
                    metrics=['accuracy'])
    model = fit_augmented_data(model, self.x_train, self.y_train, 64, 30,
                               callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)] 
                              )
    print("Test accuracy:", test_model(model, self.x_train, self.x_test, self.y_test, verbose=0))
  
  def BaseNet2_2(self):
    model = keras.Sequential()
  
    model.add(layers.Conv2D(256, kernel_size=3, input_shape=(32, 32, 3), padding='same', name='Conv2D_1'))
    model.add(layers.BatchNormalization(momentum=0.99, name='BatchNormalization_1'))
    model.add(layers.Activation(keras.activations.relu, name='Activation_1'))
  
    model.add(layers.Conv2D(128, kernel_size=5, strides=(2,2), padding='same', name='Conv2D_3'))
    model.add(layers.BatchNormalization(momentum=0.99, name='BatchNormalization_2'))
    model.add(layers.Activation(keras.activations.relu, name='Activation_2'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='MaxPooling_1'))
  
    model.add(layers.UpSampling2D(interpolation='bilinear'))
  
    model.add(layers.Conv2D(256, kernel_size=5, strides=(2,2), padding='same', name='Conv2D_5'))
    model.add(layers.BatchNormalization(momentum=0.99, name='BatchNormalization_3'))
    model.add(layers.Activation(keras.activations.relu, name='Activation_3'))
  
    model.add(layers.Conv2D(512, kernel_size=3, padding='same', name='Conv2D_6'))
    model.add(layers.BatchNormalization(momentum=0.99, name='BatchNormalization_4'))
    model.add(layers.Activation(keras.activations.relu, name='Activation_4'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='MaxPooling_2'))
  
    model.add(layers.Flatten(name='Flatten'))
    model.add(layers.Dropout(0.4, name='Dropout_0_4'))
    model.add(layers.Dense(25, activation='softmax', name='DenseSoftmax'))
    return model 
  
  def train_test_basenet2_2(self):
    model = self.BaseNet2_2()
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=self.optimizer,
                    metrics=['accuracy'])
    model = fit_augmented_data(model, self.x_train, self.y_train, 64, 30,
                               callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)] 
                              )
    print("Test accuracy:", test_model(model, self.x_train, self.x_test, self.y_test, verbose=0))

  def BaseNet2_3(self):
    inputs = keras.Input(shape=(32, 32, 3), name='Input')
    x = layers.Conv2D(256, 3, name='Conv2D_1')(inputs)
    x = layers.BatchNormalization(momentum=0.99, name='BatchNorm_1')(x)
    x = layers.Activation(keras.activations.relu, name='Activation_1')(x)
  
    x = layers.Conv2D(256, 5, name='Conv2D_2')(x)
    x = layers.BatchNormalization(momentum=0.99, name='BatchNorm_2')(x)
    x = layers.Activation(keras.activations.relu, name='Activation_2')(x)
    block_output = layers.MaxPooling2D(pool_size=(2,2), name='MaxPooling_1')(x)
  
    for i in range(2):
      x = layers.Conv2D(64, kernel_size=1, name='Conv2D_1x1_1_block_{}'.format(i+1))(x)
      x = layers.Activation(keras.activations.relu, name='Activation_1_block_{}'.format(i+1))(x)
  
      x = layers.Conv2D(64, kernel_size=3, padding="same", name='Conv2D_3x3_block_{}'.format(i+1))(block_output)
      x = layers.BatchNormalization(momentum=0.99, name='BatchNorm_1_block_{}'.format(i+1))(x)
      x = layers.Activation(keras.activations.relu, name='Activation_2_block_{}'.format(i+1))(x)
  
      x = layers.Conv2D(256, kernel_size=1, name='Conv2D_1x1_2_block_{}'.format(i+1))(x)
      x = layers.Activation(keras.activations.relu, name='Activation_3_block_{}'.format(i+1))(x)
  
      block_output = layers.add([x, block_output], name='Suma_block_{}'.format(i+1))
  
    x = layers.UpSampling2D(interpolation='bilinear', name='UpSampling')(block_output)
  
    x = layers.Conv2D(128, kernel_size=5, strides=(2,2), name='Conv2D_3')(block_output)
    x = layers.BatchNormalization(momentum=0.99, name='BatchNorm_3')(x)
    x = layers.Activation(keras.activations.relu, name='Activation_3')(x)
    x = layers.MaxPooling2D(pool_size=(4,4), name='MaxPooling_2')(x)
  
    x = layers.Flatten(name='Flatten')(x)
    x = layers.Dropout(0.4, name='Dropout_0_4')(x)
    outputs = layers.Dense(25, activation='softmax', name='DenseSoftmax')(x)
  
    model = keras.Model(inputs, outputs, name="BaseNet2_3")
    return model
  
  def train_test_basenet2_3(self):
    model = self.BaseNet2_3()
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=self.optimizer,
                    metrics=['accuracy'])
    model = fit_augmented_data(model, self.x_train, self.y_train, 64, 30,
                               callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)] 
                              )
    print("Test accuracy:", test_model(model, self.x_train, self.x_test, self.y_test, verbose=0))
  
  def apartado2(self):
    ## Optimizador elegido en el apartado 1
    self.optimizer = keras.optimizers.RMSprop()
    self.train_test_basenet2_1()
    esperar()
    self.train_test_basenet2_2()
    esperar()
    self.train_test_basenet2_3()
    esperar()
    
  """
  Funciones para el apartado 3
  """
  def cargaCUB200(self):
    self.x_train , self.y_train , self.x_test , self.y_test = cargarDatos("./imagenes")
    
    #Guardar imágenes en binario
    #np.save('train_images.npy', x_train)
    #np.save('train_labels.npy', y_train)
    #np.save('test_images.npy', x_test)
    #np.save('test_labels.npy', y_test)
    #Leer imágenes desde binario
    #y_train = np.load('train_labels.npy')
    #x_train = np.load('train_images.npy')
    #x_test = np.load('test_images.npy')
    #y_test = np.load('test_labels.npy')
    
    #Barajado de las imágenes
    from random import shuffle
    ind_train = list(range(len(self.x_train)))
    ind_test = list(range(len(self.x_test)))
    shuffle(ind_train)
    shuffle(ind_test)
    
    self.x_train = self.x_train[ind_train]
    self.y_train = self.y_train[ind_train]
    self.x_test = self.x_test[ind_test]
    self.y_test = self.y_test[ind_test]
    
    self.x_train_red, self.x_validation, self.y_train_red, self.y_validation = train_test_split(self.x_train, self.y_train, test_size=0.1)
    
    # Definir un objeto de la clase ImageDataGenerator para train y otro para test
    # con sus respectivos argumentos.
    self.datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
        )
    self.datagen.fit(self.x_train_red)
    
    self.datagen_test = ImageDataGenerator(
        preprocessing_function=preprocess_input
        )
    self.datagen_test.fit(self.x_train_red)
    
    self.aug_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        zoom_range=0.3,
        rotation_range=100,
        )
    self.aug_datagen.fit(self.x_train_red)
    
    pintaMIE(self.x_train[0:10], self.y_train[0:10], figsize=(15,50))
    pintaMIE(self.x_test[0:10], self.y_test[0:10], figsize=(15,50))
    
  def ResNetExt1(self):
    # Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
    resnet50 = ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg')
    
    resnet50.trainable = False
    x = layers.Dense(200, activation='softmax')(resnet50.output)
    #x = layers.Activation(keras.activations.softmax)(resnet50.output)
    return keras.Model(inputs=resnet50.inputs, outputs=x)
    
  def train_test_resnetext1(self):
    model = self.ResNetExt1()
    ### Train
    epochs = 30
    batch_size = 64
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
    mostrarEvolucion(model.fit(self.datagen.flow(self.x_train_red, self.y_train_red, batch_size=batch_size),
              validation_data=self.datagen_test.flow(self.x_validation, self.y_validation, batch_size=batch_size),
              steps_per_epoch=len(self.x_train_red)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.x_validation)/batch_size,
              verbose=1
              ))
    
    ### Test
    y_pred = model.predict(
                      self.datagen_test.flow(self.x_test, batch_size=1, shuffle=False),
                      steps=len(self.x_test)
                    )
    acc = calcularAccuracy(self.y_test, y_pred)
    print('Test accuracy:', acc)
    
  def ResNetExt2(self):
    # Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
    resnet50 = ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg', input_shape=(224,224,3))
    
    resnet50.trainable = False
    x = layers.Dropout(0.5, name='Dropout1')(resnet50.output)
    x = layers.Dense(1024, activation='relu', name='Dense1024')(x)
    x = layers.Dropout(0.5, name='Dropout2')(x)
    x = layers.Dense(200, activation='softmax', name='DenseSoftmax')(x)
    return keras.Model(inputs=resnet50.input, outputs=x)
    
  def train_test_resnetext2(self):
    model = self.ResNetExt2()
    ### Train
    epochs = 30
    batch_size = 64
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
    mostrarEvolucion(model.fit(self.datagen.flow(self.x_train_red, self.y_train_red, batch_size=batch_size),
              validation_data=self.datagen_test.flow(self.x_validation, self.y_validation, batch_size=batch_size),
              steps_per_epoch=len(self.x_train_red)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.x_validation)/batch_size,
              verbose=1
              ))
    
    ### Test
    y_pred = model.predict(
                      self.datagen_test.flow(self.x_test, batch_size=1, shuffle=False),
                      steps=len(self.x_test)
                    )
    acc = calcularAccuracy(self.y_test, y_pred)
    print('Test accuracy:', acc)
    
  def ResNetExt3(self):
    # Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
    resnet50 = ResNet50(include_top = False, weights = 'imagenet', pooling = None, input_shape=(224,224,3))
  
    resnet50.trainable = False
    x = layers.Conv2D(256, 3, name='Conv2D_1', padding='same')(resnet50.output)
    x = layers.BatchNormalization(momentum=0.99, name='BatchNorm_1')(x)
    x = layers.Activation(keras.activations.relu, name='Activation_1')(x)
  
    x = layers.Conv2D(256, 5, name='Conv2D_2', padding='same')(x)
    x = layers.BatchNormalization(momentum=0.99, name='BatchNorm_2')(x)
    x = layers.Activation(keras.activations.relu, name='Activation_2')(x)
    x = layers.GlobalMaxPooling2D()(x)
  
    x = layers.Dropout(0.5, name='Dropout1')(x)
    x = layers.Dense(200, activation='softmax', name='DenseSoftmax')(x)
    return keras.Model(inputs=resnet50.input, outputs=x)
  
  def train_test_resnetext3(self):
    model = self.ResNetExt3()
    ### Train
    epochs = 30
    batch_size = 64
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
    mostrarEvolucion(model.fit(self.datagen.flow(self.x_train_red, self.y_train_red, batch_size=batch_size),
              validation_data=self.datagen_test.flow(self.x_validation, self.y_validation, batch_size=batch_size),
              steps_per_epoch=len(self.x_train_red)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.x_validation)/batch_size,
              verbose=1
              ))
    
    ### Test
    y_pred = model.predict(
                      self.datagen_test.flow(self.x_test, batch_size=1, shuffle=False),
                      steps=len(self.x_test)
                    )
    acc = calcularAccuracy(self.y_test, y_pred)
    print('Test accuracy:', acc)
    
  def aug_train_test_resnetext3(self):
    model = self.ResNetExt3()
    ### Train
    epochs = 30
    batch_size = 64
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
    mostrarEvolucion(model.fit(self.aug_datagen.flow(self.x_train_red, self.y_train_red, batch_size=batch_size),
              validation_data=self.datagen_test.flow(self.x_validation, self.y_validation, batch_size=batch_size),
              steps_per_epoch=len(self.x_train_red)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.x_validation)/batch_size,
              verbose=1
              ))
    
    ### Test
    y_pred = model.predict(
                      self.datagen_test.flow(self.x_test, batch_size=1, shuffle=False),
                      steps=len(self.x_test)
                    )
    acc = calcularAccuracy(self.y_test, y_pred)
    print('Test accuracy:', acc)
    
  def fine_tuning_resnet(self):
    # Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
    resnet50 = ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg')
    resnet50.trainable = False
    x = layers.Dense(200, activation='softmax')(resnet50.output)
    model = keras.Model(inputs=resnet50.inputs, outputs=x)
    # Entrenamiento de la salida
    epochs = 30
    batch_size = 64
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
    model.fit(self.datagen.flow(self.x_train_red, self.y_train_red, batch_size=batch_size),
              validation_data=self.datagen_test.flow(self.x_validation, self.y_validation, batch_size=batch_size),
              steps_per_epoch=len(self.x_train_red)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.x_validation)/batch_size,
              verbose=1
              )
    # Ajuste del resto de la red
    resnet50.trainable = True
    epochs = 10
    batch_size = 32
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(1e-5),
                    metrics=['accuracy'])
    mostrarEvolucion(model.fit(self.datagen.flow(self.x_train_red, self.y_train_red, batch_size=batch_size),
              validation_data=self.datagen_test.flow(self.x_validation, self.y_validation, batch_size=batch_size),
              steps_per_epoch=len(self.x_train_red)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.x_validation)/batch_size,
              verbose=1
              ))
    # Test
    y_pred = model.predict(
                      self.datagen_test.flow(self.x_test, batch_size=1, shuffle=False),
                      steps=len(self.x_test)
                    )
    acc = calcularAccuracy(self.y_test, y_pred)
    print('Test accuracy:', acc)
    
  def mejora_resnetext3(self):
    # Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
    resnet50 = ResNet50(include_top = False, weights = 'imagenet', pooling = None, input_shape=(224,224,3))
  
    resnet50.trainable = False
    x = layers.Conv2D(256, 3, name='Conv2D_1', padding='same')(resnet50.output)
    x = layers.BatchNormalization(momentum=0.99, name='BatchNorm_1')(x)
    x = layers.Activation(keras.activations.relu, name='Activation_1')(x)
  
    x = layers.Conv2D(256, 5, name='Conv2D_2', padding='same')(x)
    x = layers.BatchNormalization(momentum=0.99, name='BatchNorm_2')(x)
    x = layers.Activation(keras.activations.relu, name='Activation_2')(x)
    x = layers.GlobalMaxPooling2D()(x)
  
    x = layers.Dropout(0.5, name='Dropout1')(x)
    x = layers.Dense(200, activation='softmax', name='DenseSoftmax')(x)
    model = keras.Model(inputs=resnet50.input, outputs=x)
    
    #Entrenamos solo la salida
    epochs = 30
    batch_size = 64
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
    model.fit(self.datagen.flow(self.x_train_red, self.y_train_red, batch_size=batch_size),
              validation_data=self.datagen_test.flow(self.x_validation, self.y_validation, batch_size=batch_size),
              steps_per_epoch=len(self.x_train_red)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.x_validation)/batch_size,
              verbose=1
              )
  
    # Ajuste fino
    resnet50.trainable = True
    epochs = 10
    batch_size = 32
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(1e-5),
                    metrics=['accuracy'])
    mostrarEvolucion(model.fit(self.datagen.flow(self.x_train_red, self.y_train_red, batch_size=batch_size),
              validation_data=self.datagen_test.flow(self.x_validation, self.y_validation, batch_size=batch_size),
              steps_per_epoch=len(self.x_train_red)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.x_validation)/batch_size,
              verbose=1
              ))
    
    y_pred = model.predict(
                      self.datagen_test.flow(self.x_test, batch_size=1, shuffle=False),
                      steps=len(self.x_test)
                    )
    acc = calcularAccuracy(self.y_test, y_pred)
    print('Test accuracy:', acc)
    
  def apartado3(self):
    self.cargaCUB200()
    esperar()
    self.train_test_resnetext1()
    esperar()
    self.train_test_resnetext2()
    esperar()
    self.train_test_resnetext3()
    esperar()
    self.fine_tuning_resnet()
    esperar()
    self.mejora_resnetext3()
    esperar()
    
    
  """
  Funciones para el bonus
  """
  def cargarPathMNIST(self):
    dir = "pathmnist"
    
    train_images = np.load("./imagenes/"+dir+"/train_images.npy")
    train_labels = np.load("./imagenes/"+dir+"/train_labels.npy")
    val_images = np.load("./imagenes/"+dir+"/val_images.npy")
    val_labels = np.load("./imagenes/"+dir+"/val_labels.npy")
    test_images = np.load("./imagenes/"+dir+"/test_images.npy")
    test_labels = np.load("./imagenes/"+dir+"/test_labels.npy")
    
    #Barajado y reducción de las imágenes 
    from random import shuffle
    ind_train = list(range(len(train_images)))
    ind_val = list(range(len(val_images)))
    ind_test = list(range(len(test_images)))
    shuffle(ind_train)
    shuffle(ind_val)
    shuffle(ind_test)
    
    train_images = train_images[ind_train][0:int(len(train_images))]
    self.train_labels = train_labels[ind_train,0][0:int(len(train_labels))]
    val_images = val_images[ind_val][0:int(len(val_images))]
    self.val_labels = val_labels[ind_val,0][0:int(len(val_labels))]
    test_images = test_images[ind_test][0:int(len(test_images))]
    self.test_labels = test_labels[ind_test,0][0:int(len(test_labels))]
    
    #Para las más grandes reducimos el número para no quedarnos sin memoria
    self.size_train = int(len(train_images)/10)
    self.b_train_images = np.asarray([cv.resize(x, dsize=(224,224)) for x in train_images[:self.size_train]])
    self.b_val_images = np.asarray([cv.resize(x, dsize=(224,224)) for x in val_images])
    self.b_test_images = np.asarray([cv.resize(x, dsize=(224,224)) for x in test_images])
    
    pintaMIE(self.b_train_images[0:10], train_labels[0:10], figsize=(10,30))
    
  def extrac_car_nasnetmobile(self):
    from keras.applications.nasnet import NASNetMobile, preprocess_input
    #Preprocesadores de los datos
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )
    datagen.fit(self.b_train_images)
    
    datagen_test = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )
    datagen_test.fit(self.b_train_images)
    
    ##Definición del modelo (cambiamos solo la salida de NASNetMobile)
    #Input shape obligatoria al usar los pesos de imagenet
    base_model = NASNetMobile(input_shape=(224,224,3), include_top=False, weights='imagenet', pooling='avg')
    base_model.trainable = False
    x = layers.Dense(9, activation='softmax', name='DenseSoftmax')(base_model.output)
    model = keras.Model(inputs=base_model.input, outputs=x)
    ##Entrenamiento de la salida
    epochs = 100
    batch_size = 32
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    mostrarEvolucion(model.fit(datagen.flow(self.b_train_images, self.train_labels[:self.size_train], batch_size=batch_size),
              validation_data=datagen_test.flow(self.b_val_images, self.val_labels, batch_size=batch_size),
              steps_per_epoch=len(self.b_train_images)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              validation_steps=len(self.b_val_images)/batch_size,
              verbose=1
              ))
    y_pred = model.predict(
                      datagen_test.flow(self.b_test_images, batch_size=1, shuffle=False),
                      steps=len(self.b_test_images)
                    )
    acc = calcularAccuracy(self.test_labels, y_pred)
    print('Test accuracy:', acc)
    
    ### Repetimos el entrenamiento anterior, pero entrenando también las últimas 
    ### 50 capas con un lr más bajo
    #Descongelamiento de las 50 últimas capas
    base_model.trainable = True
    for layer in base_model.layers[:720]:
      layer.trainable = False
    for layer in base_model.layers[720:]:
      layer.trainable = True
    # Incluimos aumento de datos
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        zoom_range=0.3,
        rotation_range=50,
        )
    datagen.fit(self.b_train_images)
    datagen_test = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )
    datagen_test.fit(self.b_train_images)
    
    #Las imágenes que tenemos no son suficientes para hacer un fine tuning adecuado
    # pero podemos aplicar un aumento de datos, puesto que ImageDataGenerator genera
    # las imágenes proceduralmente
    epochs = 20
    batch_size = 32
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(8e-5),
                  metrics=['accuracy'])
    mostrarEvolucion(model.fit(datagen.flow(self.b_train_images, self.train_labels[:self.size_train], batch_size=batch_size),
              validation_data=datagen_test.flow(self.b_val_images, self.val_labels, batch_size=batch_size),
              steps_per_epoch=len(self.b_train_images)/batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
              validation_steps=len(self.b_val_images)/batch_size,
              verbose=1
              ))
    
    y_pred = model.predict(
                      datagen_test.flow(self.b_test_images, batch_size=1, shuffle=False),
                      steps=len(self.b_test_images)
                    )
    acc = calcularAccuracy(self.test_labels, y_pred)
    print('Test accuracy:', acc)
    
  def bonus(self):
    self.cargarPathMNIST()
    esperar()
    self.extrac_car_nasnetmobile()
    esperar()
    

def main():
  p2 = P2()
  p2.apartado1()
  p2.apartado2()
  p2.apartado3()
  p2.bonus()
  
if __name__ == "__main__":
    main()