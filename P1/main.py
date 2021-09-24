#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alejandro Alonso Membrilla
"""

"""
Funciones de uso general
"""
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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

#Normaliza y pinta en pantalla un conjunto de imágenes con sus 
# respectivos títulos
def pintaMIE(vim, titulos, ncol = 2, figsize=(12,12), winname=None):
  import matplotlib.pyplot as plt
  from math import ceil
  fig = plt.figure(num=winname, figsize=figsize)
  for k, im in enumerate(vim):
    sub = fig.add_subplot( ceil(len(vim)/ncol), ncol, k + 1 )
    plt.axis('off')
    plt.title(titulos[k])
    cmap = 'gray' if len(im.shape) == 2 else None
    sub.imshow(normalizar_matriz(im), cmap=cmap)
  plt.show(block=False)
  
#Establece un punto de parada
def esperar():
  input("Pulsa una tecla para continuar...")
  plt.close('all')
  
"""
Funciones para el ejercicio 1A
"""
#Cuando se calcula una máscara gaussiana o sus derivadas a partir de un tamaño
# de máscara dado, sirve para calcular el valor de la desviación típica a partir 
# de dicho tamaño
def calc_sigma(diam):
  return ((diam-1)//2 )/3

#La máscara es la gaussiana discretizada
def masc_gaussiana(sigma=None, diam=None):
  if sigma == None:
    sigma = calc_sigma(diam)
    lim = diam//2
  else:
    lim = math.ceil(3*sigma)  #radio de la máscara
  rango = np.arange(-lim, lim+1)
  masc = np.asarray( [math.exp(-(x**2)/(2*sigma**2)) for x in rango] )
  return masc/sum(masc)

#La máscara es la derivada de la gaussiana discretizada
def masc_derivada1_gauss(sigma=None, diam=None):
  if sigma == None:
    sigma = calc_sigma(diam)
    lim = diam//2
  else:
    lim = math.ceil(3*sigma)
  rango = np.arange(-lim, lim+1)
  masc = np.asarray( [math.exp(-(x**2)/(2*sigma**2)) for x in rango] )
  masc *= -rango/(sigma**2)
  return masc   #En este caso no normalizamos la máscara porque su suma debe valer 0

#La máscara es la segunda derivada de la gaussiana discretizada
def masc_derivada2_gauss(sigma=None, diam=None):
  if sigma == None:
    sigma = calc_sigma(diam)
    lim = diam//2
  else:
    lim = math.ceil(3*sigma)
  rango = np.arange(-lim, lim+1)
  sig2 = sigma*sigma
  masc = np.asarray( [math.exp(-(x**2)/(2*sig2)) for x in rango] )
  masc *= (rango**2-sig2)/(sig2*sig2)
  return masc   #En este caso no normalizamos la máscara porque su suma debe valer 0

def ejercicio1A():
  #Mostramos gráficas de la gaussiana y sus derivadas discretizadas
  sigma = 3.0
  plt.plot(masc_gaussiana(sigma))
  plt.plot(masc_derivada1_gauss(sigma))
  plt.plot(masc_derivada2_gauss(sigma))
  plt.axis("off")
  plt.legend(["Gaussiana", "Derivada 1ª", "Derivada 2ª"])
  plt.show(block=False)
  esperar()

"""
Funciones para el ejercicio 1B
"""
#El método asume que el padding solo debe hacerse en los márgenes laterales,
#porque únicamente se utiliza en correlaciones 1D con máscaras aplicadas por filas
#modo indica el tipo de padding: 
#    1. "negro" añade márgenes negros
#    2. "ext" añade márgenes replicados
#    3. "ciclo" añade márgenes en bucle (mosaico) 
#El proceso consiste en generar el marco a la izquierda y a la derecha, y a 
# concatenarlos con la imagen original
def pad_image( image, pad_size, modo="ciclo" ):
  nfilas = len(image)
  
  if modo == "negro":
    #Padding a izquierda y derecha
    padding = np.zeros((nfilas, pad_size))
    padded_image = np.concatenate( (padding, image, padding), axis=1 )

  elif modo == "ext":
    #Padding a izquierda y derecha
    padding_izda = np.reshape( np.repeat(image[:,0], pad_size), (nfilas, pad_size) )
    padding_dcha = np.reshape( np.repeat(image[:,-1], pad_size), (nfilas, pad_size) )
    padded_image = np.concatenate( (padding_izda, image, padding_dcha), axis=1 )

  elif modo == "ciclo":
    #Padding a izquierda y derecha
    padding_izda = np.reshape( image[:,-pad_size-1:-1], (nfilas, pad_size) )
    padding_dcha = np.reshape( image[:,0:pad_size], (nfilas, pad_size) )
    padded_image = np.concatenate( (padding_izda, image, padding_dcha), axis=1 )

  return padded_image

#Aplica la correlación en todos los elementos de una columna
def corr_en_columna(image, mask, i, radio_mask):
  return np.sum(mask*image[:,i-(radio_mask+1):i+radio_mask], axis=1 )

#Aplica la correlación 1D por filas en todos los elementos de una matriz
def correlation1D(image, mask, modo="ciclo"):
  radio_mask = math.floor( len(mask)/2 )
  padded_image = pad_image(image, radio_mask, modo=modo)
  image_ncol = np.size(padded_image, axis=1)
  correlacion = np.asarray( 
      [ corr_en_columna(padded_image, mask, i, radio_mask) for i in range(radio_mask+1, image_ncol-radio_mask+1) ] 
    ).T
  #Trasponemos porque np.asarray convierte los elementos de la lista en filas,
  # pero son columnas
  return correlacion

#Aplica correlación 2D separable (una máscara horizontal y luego otra vertical)  
def correlation2D(image, hmask, vmask, modo="ciclo"):
  #Si un filtro es separable, basta con aplicar la correlación unidimensional 2 veces:
  #Una por filas
  correlation_filas = correlation1D(image, hmask, modo=modo)
  #Y otra por columnas
  correlation = correlation1D(correlation_filas.T, vmask, modo=modo).T
  return correlation

#Devuelve una convolución 2D mediante una correlación con la máscara invertida
def convolution2D(image, hmask, vmask, modo="ciclo"):
  return correlation2D(image, hmask[::-1], vmask[::-1], modo=modo)

#NOTA: esta función incluye código usado únicamente en el Bonus
# (el bool monobanda indica que la imagen está en escala de grises)
def suavizar(img, sigma, modo="ciclo"):
  masc_gauss = masc_gaussiana(sigma)
  monobanda = len(img.shape) == 2
  if monobanda:
    return convolution2D(img, masc_gauss, masc_gauss, modo=modo)
  else:
    #Calculamos la convolución 2D de cada banda y le añadimos una nueva dimensión...
    banda_b = convolution2D(img[:,:,0], masc_gauss, masc_gauss, modo=modo)[:,:,np.newaxis]
    banda_g = convolution2D(img[:,:,1], masc_gauss, masc_gauss, modo=modo)[:,:,np.newaxis]
    banda_r = convolution2D(img[:,:,2], masc_gauss, masc_gauss, modo=modo)[:,:,np.newaxis]
    #...para poder concatenar las bandas sin problemas
    return np.concatenate((banda_b, banda_g, banda_r), axis=2)

def ejercicio1B(sigma=3.0):
  img = cv.imread('imagenes/cat.bmp', 0)
  blurcv = cv.GaussianBlur(img, (0,0), sigmaX=sigma)
  blur = suavizar(img, sigma)
  pintaMIE([img, blurcv, blur], ["Original", "Suavizada con OpenCV", "Suavizado propio"], ncol=3)
  esperar()
  

"""
Funciones del apartado 1C
"""
#Devuelve el array arr centrado en otro array de tamaño size
def centrar(arr, size):
  if size == arr.size:
    return arr
  diff = size - arr.size
  diff_half = int(diff/2)
  if diff > 0:
    zeros = np.zeros(diff_half)
    return np.concatenate( (zeros, arr, zeros) )
  else:
    return np.array(arr[-diff_half:diff_half])

def ejercicio1C():
  #Mostramos gráficas comparando la derivada de la gaussiana discretizada con
  # el resultado de getDerivKernels, para varios sigmas y tamaños de bloque
  for orden in [1,2]:
    for bsize in [5,7]:  
      fig = plt.figure(figsize=(13,4))
      for sigma in [1,1.2]:
        #OpenCV da el kernel de la 1ª derivada invertido, y en forma de array
        # multidimensional, de ahí viene la siguiente indexación
        deriv = np.transpose(cv.getDerivKernels(orden,orden,5)[0])[0][::-1]
        masc = [masc_derivada1_gauss, masc_derivada2_gauss][orden-1](sigma=1)
        fig.add_subplot(1, 2, 1 if sigma == 1 else 2)
        plt.plot(np.arange(masc.size), masc, centrar(deriv, masc.size))
        plt.title("Derivada de orden {}. Tamaño de bloque {}. Sigma={}.".format(orden, bsize, sigma))
        plt.legend(["Derivada Gaussiana", "getDerivKernels"])
      plt.show(block=False)

  esperar()
  
  #Escalamos las máscaras y volvemos a compararlas con los kernels
  fig = plt.figure(figsize=(13,8))

  deriv = np.transpose(cv.getDerivKernels(1,1,5)[0])[0][::-1]
  masc = masc_derivada1_gauss(sigma=1)*3.3
  fig.add_subplot(2, 2, 1)
  plt.plot(np.arange(masc.size), masc, centrar(deriv, masc.size))
  plt.title("Derivada de orden 1. Tamaño de bloque 5. Sigma=1.")
  plt.legend(["Derivada Gaussiana", "getDerivKernels"])
  
  deriv = np.transpose(cv.getDerivKernels(1,1,7)[0])[0][::-1]
  masc = masc_derivada1_gauss(sigma=1.2)*11
  fig.add_subplot(2, 2, 2)
  plt.plot(np.arange(masc.size), masc, centrar(deriv, masc.size))
  plt.title("Derivada de orden 1. Tamaño de bloque 7. Sigma=1.2.")
  plt.legend(["Derivada Gaussiana", "getDerivKernels"])
  
  deriv = np.transpose(cv.getDerivKernels(2,2,5)[0])[0][::-1]
  masc = masc_derivada2_gauss(sigma=1)*2
  fig.add_subplot(2, 2, 3)
  plt.plot(np.arange(masc.size), masc, centrar(deriv, masc.size))
  plt.title("Derivada de orden 2. Tamaño de bloque 5. Sigma=1.")
  plt.legend(["Derivada Gaussiana", "getDerivKernels"])
  
  deriv = np.transpose(cv.getDerivKernels(2,2,7)[0])[0][::-1]
  masc = masc_derivada2_gauss(sigma=1.2)*6
  fig.add_subplot(2, 2, 4)
  plt.plot(np.arange(masc.size), masc, centrar(deriv, masc.size))
  plt.title("Derivada de orden 2. Tamaño de bloque 7. Sigma=1.2.")
  plt.legend(["Derivada Gaussiana", "getDerivKernels"])
  
  plt.show(block=False)
  
  esperar()
  
"""Funciones del ejercicio 1D"""
def LoG(image, sigma, modo="ext"):
  #Calculamos ambas máscaras a utilizar
  m_gauss = masc_gaussiana(sigma)
  m_deriv2_gauss = masc_derivada2_gauss(sigma)
  #Hacemos las convoluciones para obtener las derivadas
  img_deriv2_x = convolution2D(image, m_deriv2_gauss, m_gauss, modo=modo)
  img_deriv2_y = convolution2D(image, m_gauss, m_deriv2_gauss, modo=modo)

  #Obtenemos el resultado
  lapl_of_gauss = img_deriv2_x + img_deriv2_y
  norm_LoG = sigma**2*lapl_of_gauss
  #norm_LoG = normalizar_matriz(sigma**2*lapl_of_gauss)
  return norm_LoG

#Ejemplo de cálculo de LoG (se muestran las derivadas parciales y las máscaras)
def ejemplo_LoG(sigma, modo):
  img = cv.imread('imagenes/cat.bmp', 0)
  
  img_deriv2_x = np.copy(img)
  img_deriv2_y = np.copy(img)
  
  #Calculamos ambas máscaras a utilizar
  m_gauss = masc_gaussiana(sigma)
  m_deriv2_gauss = masc_derivada2_gauss(sigma)
  #Hacemos las convoluciones para obtener las derivadas
  img_deriv2_x = convolution2D(img_deriv2_x, m_deriv2_gauss, m_gauss, modo=modo)
  img_deriv2_y = convolution2D(img_deriv2_y, m_gauss, m_deriv2_gauss, modo=modo)
  
  #Pintamos las derivadas...
  pintaMIE( [img_deriv2_x,img_deriv2_y], ['Derivada con respecto de x', 'Derivada con respecto de y'], 2 )
  #...y las máscaras
  mascaras = [ np.outer(m_deriv2_gauss, m_gauss), np.outer(m_gauss, m_deriv2_gauss) ]
  pintaMIE ( mascaras, ['Máscara gaussiana', 'Derivada segunda de la gaussiana'] )
  
  #Podríamos haber utilizado los cálculos anteriores en vez de llamar a la 
  # función LoG, pero de esta forma vemos que esta funciona correctamente
  pintaMIE( [LoG(img, 1, modo='negro')], ["Laplaciana de la gaussiana"], ncol=1 )
  esperar()

def ejercicio1D():
  ###Ejemplo de cálculo de LoG (se muestran las derivadas parciales y las máscaras)
  ejemplo_LoG(1, "negro")
  ejemplo_LoG(3, "ext")
  ejemplo_LoG(1, "negro")
  ejemplo_LoG(3, "ext")

"""
Funciones para el ejercicio 2A
"""
#Devuelve otro array con las filas y columnas pares de la imagen original
def downsample_v0(img):
  return img[::2,::2]

def pyrDown_v0(img, sigma, modo='ciclo'):
  return downsample_v0( suavizar(img, sigma, modo=modo) )

#Suaviza y reduce el tamaño de la imagen a la mitad
def pyrDown(img, sigma, modo='ciclo'):
  return cv.resize(suavizar(img, sigma, modo), (img.shape[1]//2, img.shape[0]//2))

#Devuelve la pirámide gaussiana de una imagen dada
def piramide_gaussiana(img, sigma=1.0, altura=4, modo="ciclo"):
  piramide_suavizada = [img]
  for i in range(altura):
    piramide_suavizada.append( pyrDown(piramide_suavizada[i], sigma, modo) )
  return piramide_suavizada

def ejercicio2A():
  img = cv.imread('imagenes/cat.bmp', 0)
  #Sin suavizar
  piramide = [img]
  for i in range(4):
    piramide.append( downsample_v0(piramide[i]) )
  
  titulos = [ "Nivel {}".format(i) for i in range(5) ]
  pintaMIE(piramide, titulos, ncol=5, figsize=(20,20), winname="Pirámide sin suavizar")
  
  #Suavizadas
  for sigma in [0.9, 1.0, 1.1, 1.2]:
    piramide_suavizada = piramide_gaussiana(img, sigma, altura=4, modo="ext")
    pintaMIE(piramide_suavizada, titulos, ncol=5, figsize=(20,20), winname="Pirámide suavizada. Sigma={}".format(sigma))
    
  esperar()
  
  #No muestran mucha diferencia entre ellas. Probamos con desviaciones típicas más diferenciadas
  for sigma in [0.5, 1, 2, 3]:
    piramide_suavizada = piramide_gaussiana(img, sigma, altura=4, modo="ext")
    pintaMIE(piramide_suavizada, titulos, ncol=5, figsize=(20,20), winname="Pirámide suavizada. Sigma={}".format(sigma))
    
  esperar()
  
"""
Funciones para el ejercicio 2B
"""
#Inserta ceros entre cada dos columnas y dos filas
def upsample_v0(img):
  upsam = np.zeros(shape=(np.size(img,0)*2, np.size(img,1)*2))
  upsam[::2,::2] = img
  return upsam

#Suaviza la imagen aumentada con una máscara incrementada. Equivale a suavizar y
# multiplicar la imagen por 4 (análogo al método utilizado por OpenCV)
def pyrUp_v0(img, sigma, modo="ciclo"):
  masc = masc_gaussiana(sigma)*2
  return convolution2D(upsample_v0(img), masc, masc, modo)

#Aumenta la imagen usando cv.resize con interpolación bilineal (por defecto)
def upsample(img):
  return cv.resize(img, (np.size(img,1)*2, np.size(img,0)*2))

#Aumenta la imagen y la suaviza
def pyrUp(img, sigma, modo="ciclo"):
  return suavizar(upsample(img), sigma, modo)

#Si son del mismo tamaño, las resta. Si no, cambia de tamaño (interpolación bilineal) la segunda
# al tamaño de la primera y entonces las resta
def restar_imgs(img1, img2):
  if img1.shape == img2.shape:
    return img1-img2
  else:
    #El parámetro dsize de cv.resize tiene la orientación contraria al atributo shape de numpy
    new_shape = (img1.shape[1], img1.shape[0])
    return img1-cv.resize( img2, new_shape )

def piramide_laplaciana(img, sigma=1.0, altura=4, modo="ciclo"):
  piramide_lapl = []
  pir_gauss = piramide_gaussiana(img, sigma, altura, modo)
  for i in range(1,altura+1):
    piramide_lapl.append( restar_imgs( pir_gauss[i-1], pyrUp(pir_gauss[i], sigma, modo) ) )
    #piramide_lapl.append( -LoG(pir_gauss[i-1], sigma=sigma) )
  return piramide_lapl

#Calcula la pirámide laplaciana usando la función pyrUp de OpenCV
def piramide_laplaciana_cv(img, sigma=1.0, altura=4, modo="ciclo"):
  piramide_lapl = []
  pir_gauss = piramide_gaussiana(img, sigma, altura, modo)
  for i in range(1,altura+1):
    piramide_lapl.append( restar_imgs( pir_gauss[i-1], cv.pyrUp(pir_gauss[i]) ) )
  return piramide_lapl

def ejercicio2B():
  img = cv.imread('imagenes/cat.bmp', 0)
  imagenes = [pyrUp_v0(img, 1.0, "ext"), pyrUp(img, 1.0, "ext"), cv.pyrUp(img)]
  titulos = ["PyrUp primera versión", "Segunda versión", "Versión de OpenCV"]
  pintaMIE(imagenes, titulos, ncol=3, winname="Comparación de PyrUps")
  
  esperar()
  
  titulos = [ "Nivel {}".format(i) for i in range(1, 5) ]
  img2 = cv.imread('imagenes/marilyn.bmp', 0)
  
  ##Comparamos nuestras pirámides laplacianas con las conseguidas con cv.pyrUp
  #Pirámides laplacianas con pyrUp propio
  pintaMIE( piramide_laplaciana(img, sigma=1.0, modo="ext"), titulos, 4, (15,15), "Pirámide cat.bmp" )
  pintaMIE( piramide_laplaciana(img2, sigma=1.0, modo="ext"), titulos, 4, (15,15), "Pirámide marilyn.bmp")
  #Pirámides laplacianas con cv.pyrUp
  pintaMIE( piramide_laplaciana_cv(img, sigma=1.0, modo="ext"), titulos, 4, (15,15), "Pirámide cat.bmp con cv.pyrUp" )
  pintaMIE( piramide_laplaciana_cv(img2, sigma=1.0, modo="ext"), titulos, 4, (15,15), "Pirámide marilyn.bmp con cv.pyrUp" )
  
  esperar()
  
  #Ejemplo de uso de la pirámide laplaciana: reconstrucción de imágenes
  sigma = 1.0
  im = piramide_gaussiana(img, altura=4)[4]
  pir_laplaciana = piramide_laplaciana(img, altura=4, modo='ext')
  for i in range(4,0,-1):
    shape = pir_laplaciana[i-1].shape
    #Aumentamos la imagen y la suavizamos. El cv.resize exterior se debe a que
    # la creación de pirámides gaussianas y laplacianas asume que las dimensiones
    # de la imagen son potencia de 2, pero estamos manipulando imágenes con
    # número de filas/columnas impar
    im_up = cv.resize( pyrUp(im, sigma=sigma), (shape[1], shape[0]) ) 
    im = normalizar_matriz(pir_laplaciana[i-1]) + normalizar_matriz(im_up)
    pintaMIE([pir_laplaciana[i-1], im_up, im], ["Laplaciana", "Reconstrucción anterior", "Imagen reconstruida"], ncol=3, winname="Reconstrucción de x{}".format(i))
  pintaMIE([img, im], ["Original", "Reconstruida"], winname="Comparación entre la original y la reconstrucción")
  
  esperar()

"""
Funciones para el ejercicio 3A
"""
"""
El 3A se ha hecho de forma conjunta con el 3C, mostrando varias parejas y sus 
imágenes híbridas para varios valores de sigmas y eligiendo los que obtenían mejores resultados.
"""
def ejercicio3A():
  pass

"""
Funciones para el ejercicio 3B
"""
def high_img(img, sigma, modo='ciclo'):
  #k = 1.5
  return img-suavizar(img, sigma, modo)
  #Alternativamente
  #return LoG(img, sigma, modo)

def low_img(img, sigma, modo='ciclo'):
  return suavizar(img, sigma, modo)

def hybrid(himg, limg, hsigma, lsigma, modo='ciclo'):
  return high_img(himg, hsigma, modo=modo) + low_img(limg, lsigma, modo=modo)

def pintaHybrid(himg, limg, hsigma, lsigma, modo='ciclo', figsize=(9,9)):
  high = high_img(himg, hsigma, modo)
  low = low_img(limg, lsigma, modo)
  imgs = [ high, low, high + low ]
  titulos = [ "Frecuencias altas", "Frecuencias bajas", "Imagen híbrida" ]
  pintaMIE(imgs, titulos, ncol=3)
  
def ejercicio3B():
  hsigma = 0.8
  lsigma = 7.0
  modo='ciclo'
  himg = cv.imread('imagenes/einstein.bmp', 0)
  limg = cv.imread('imagenes/marilyn.bmp', 0)
  pintaHybrid(himg, limg, hsigma, lsigma, modo)
  esperar()
  
"""Funciones para el ejercicio 3C"""
#Mostramos por pantalla las imágenes híbridas para cada pareja,
# comparando los resultados para diversos valores de sigma (ejercicio 3A)
def comp_sigmas_pajaro_avion():
  hsigmas = (1.0, 1.5, 2.0)
  lsigmas = (5.0, 8.0, 11.0)
  modo='ciclo'
  himg = cv.imread('imagenes/bird.bmp', 0)
  limg = cv.imread('imagenes/plane.bmp', 0)
  
  for hsigma, lsigma in zip(hsigmas, lsigmas):
    pintaHybrid(himg, limg, hsigma, lsigma, modo)
    
  esperar()
  
def comp_sigmas_bici_moto():
  hsigmas = (0.6, 0.8, 1.4)
  lsigmas = (7.0, 9.0, 11.0)
  modo='ciclo'
  himg = cv.imread('imagenes/bicycle.bmp', 0)
  limg = cv.imread('imagenes/motorcycle.bmp', 0)
  
  for hsigma, lsigma in zip(hsigmas, lsigmas):
    pintaHybrid(himg, limg, hsigma, lsigma, modo)
    
  esperar()
  
def comp_sigmas_perro_gato():
  hsigmas = (2.0, 2.8, 3.6)
  lsigmas = (10.0, 12.0, 14.0)
  modo='ciclo'
  himg = cv.imread('imagenes/dog.bmp', 0)
  limg = cv.imread('imagenes/cat.bmp', 0)
  
  for hsigma, lsigma in zip(hsigmas, lsigmas):
    pintaHybrid(himg, limg, hsigma, lsigma, modo)
    
  esperar()
  
def comp_sigmas_pez_submarino():
  hsigmas = (0.4, 0.7, 1.2)
  lsigmas = (5.0, 7.0, 9.0)
  modo='ciclo'
  himg = cv.imread('imagenes/fish.bmp', 0)
  limg = cv.imread('imagenes/submarine.bmp', 0)
  
  for hsigma, lsigma in zip(hsigmas, lsigmas):
    pintaHybrid(himg, limg, hsigma, lsigma, modo)
    
  esperar()
  
def comp_sigmas_einstein_marilyn():
  hsigmas = (0.4, 0.8, 1.3)
  lsigmas = (5.0, 7.0, 9.0)
  modo='ciclo'
  himg = cv.imread('imagenes/einstein.bmp', 0)
  limg = cv.imread('imagenes/marilyn.bmp', 0)
  
  for hsigma, lsigma in zip(hsigmas, lsigmas):
    pintaHybrid(himg, limg, hsigma, lsigma, modo)
    
  esperar()
  
def ejercicio3C():
  comp_sigmas_pajaro_avion()
  comp_sigmas_bici_moto()
  comp_sigmas_perro_gato()
  comp_sigmas_pez_submarino()
  comp_sigmas_einstein_marilyn()
  
"""
Funciones para el ejercicio 3D
"""
#Tomando los sigmas considerados óptimos para cada pareja (ejercicio 3A/3C), 
# mostramos la pirámide gaussiana de la hibridación de cada pareja
def ejercicio3D():
  titulos = [ "Nivel {}".format(i) for i in range(6) ]
  modo = 'ciclo'
  
  #Pirámide para el híbrido pájaro-avión
  hsigma = 1.5
  lsigma = 8.0
  himg = cv.imread('imagenes/bird.bmp', 0)
  limg = cv.imread('imagenes/plane.bmp', 0)
  hyb = hybrid(himg, limg, hsigma, lsigma, modo)
  pyr_gauss = piramide_gaussiana(hyb, sigma=1.0, altura=4, modo=modo)
  pintaMIE(pyr_gauss, titulos, ncol=3, figsize=(12,8))
  esperar()
  
  #Pirámide para el híbrido bici-moto
  hsigma = 0.8
  lsigma = 9.0
  himg = cv.imread('imagenes/bicycle.bmp', 0)
  limg = cv.imread('imagenes/motorcycle.bmp', 0)
  hyb = hybrid(himg, limg, hsigma, lsigma, modo)
  pyr_gauss = piramide_gaussiana(hyb, sigma=1.0, altura=4, modo=modo)
  pintaMIE(pyr_gauss, titulos, ncol=3, figsize=(12,8))
  esperar()
  
  #Pirámide para el híbrido einstein-marilyn
  hsigma = 0.8
  lsigma = 7.0
  himg = cv.imread('imagenes/einstein.bmp', 0)
  limg = cv.imread('imagenes/marilyn.bmp', 0)
  hyb = hybrid(himg, limg, hsigma, lsigma, modo)
  pyr_gauss = piramide_gaussiana(hyb, sigma=1.0, altura=4, modo=modo)
  pintaMIE(pyr_gauss, titulos, ncol=3, figsize=(12,8))
  esperar()
  
"""
Funciones para el Bonus 1
"""
def color_hyb_pajaro_avion(modo='ciclo'):
  hsigma = 2.0
  lsigma = 9.0
  himg = cv.imread('imagenes/bird.bmp', 1)[:,:,::-1]
  limg = cv.imread('imagenes/plane.bmp', 1)[:,:,::-1]
  pintaHybrid(himg, limg, hsigma, lsigma, modo=modo, figsize=())
  
def color_hyb_bici_moto(modo='ciclo'):
  hsigma = 1.0
  lsigma = 9.0
  himg = cv.imread('imagenes/bicycle.bmp', 1)[:,:,::-1]
  limg = cv.imread('imagenes/motorcycle.bmp', 1)[:,:,::-1]
  pintaHybrid(himg, limg, hsigma, lsigma, modo=modo)
  
def color_hyb_perro_gato(modo='ciclo'):
  hsigma = 3.5
  lsigma = 12.0
  himg = cv.imread('imagenes/dog.bmp', 1)[:,:,::-1]
  limg = cv.imread('imagenes/cat.bmp', 1)[:,:,::-1]
  pintaHybrid(himg, limg, hsigma, lsigma, modo=modo)
  
def color_hyb_pez_submarino(modo='ciclo'):
  hsigma = 1.8
  lsigma = 9.0
  himg = cv.imread('imagenes/fish.bmp', 1)[:,:,::-1]
  limg = cv.imread('imagenes/submarine.bmp', 1)[:,:,::-1]
  pintaHybrid(himg, limg, hsigma, lsigma, modo=modo)
  
def color_hyb_einstein_marilyn(modo='ciclo'):
  hsigma = 0.9
  lsigma = 6.0
  himg = cv.imread('imagenes/einstein.bmp', 1)[:,:,::-1]
  limg = cv.imread('imagenes/marilyn.bmp', 1)[:,:,::-1]
  pintaHybrid(himg, limg, hsigma, lsigma, modo=modo)

def bonus1():
  color_hyb_pajaro_avion()
  color_hyb_bici_moto()
  color_hyb_perro_gato()
  color_hyb_pez_submarino()
  color_hyb_einstein_marilyn()
  esperar()
  
"""
Funciones para el Bonus 2
"""
def color_hyb_erizo_bola(modo='ciclo'):
  hsigma = 2.9
  lsigma = 26.0
  modo='ciclo'
  limg = cv.imread('imagenes/bolabolos.jpeg', 1)[:,:,::-1]
  himg = cv.imread('imagenes/erizo.jpg', 1)[:,:,::-1]
  himg = cv.resize(himg, (limg.shape[1], limg.shape[0]))
  pintaHybrid(himg, limg, hsigma, lsigma, modo=modo, figsize=(15,15))

def bonus2():
  color_hyb_erizo_bola()
  esperar()
  
def main():
  ejercicio1A()
  ejercicio1B()
  ejercicio1C()
  ejercicio1D()
  
  ejercicio2A()
  ejercicio2B()
  
  ejercicio3A()
  ejercicio3B()
  ejercicio3C()
  ejercicio3D()
  
  bonus1()
  bonus2()
  
main()
