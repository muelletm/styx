"""
module documentation
"""

import numpy as np 
import imageio
import tensorflow as tf
import requests # für http

from . import checkpoint
from . import layer
from . import graph
from .ops import * # das müsste ok sein, weil operations sehr spezielle namen haben und sich da nichts in die quere kommt

from . import model
import math
import matplotlib.pyplot as plt
import skimage

from . import constants

# pfad für alle daten, standardmäßig colab
constants.datapath = "/drive/My Drive/colab/data"

def setDataPath(path):
  constants.datapath = path


# standard objekt inspect
def inspect(obj):
	for key in dir(obj):
		if(not key.startswith("_")):
			value = getattr(obj,key) 
			print(f"{key}: {type(value)}")
			if(value.__doc__ != None):
				print("\t"+value.__doc__.split("\n")[0])


# todo: könnte man auch über import * also __all__ machen
def imports(g):
	""" importiere alle standard namen. muss mit globals() aufgerufen werden:
	stupid.imports(globals())
	"""
	import tensorflow as tf
	import IPython.display as display
	import matplotlib.pyplot as plt # zeigt bilder an plt.imshow(image)
	import matplotlib as mpl
	import numpy as np
	import functools
	import tensorflow_datasets as tfds
	import matplotlib.image as mpimg
	import keras

	g["tf"] = tf 
	g["display"] = display 
	g["plt"] = plt 
	g["mpl"] = mpl 
	g["numpy"] = np 
	g["np"] = np
	g["functools"] = functools 
	g["tfds"] = tfds
	g["mpimg"] = mpimg
	g["keras"] = keras


	print("verfügbare Variablen: tf, display, plt, np/numpy, functools, tfds")


def img(filename='steine.jpg'):
	""" lade bild aus drive/colab/images
	google drive muss gemountet sein"""
	return mpimg.imread('/content/drive/My Drive/colab/images/'+filename)


def load(path_to_img):
	""" ziel: allgemeine load methode. im moment nur bilder
	img pfad -> tensor mit maxdim 512 und values 0..1
	"""
	max_dim = 512
	img = tf.io.read_file(path_to_img)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)
	img = tf.image.resize(img, new_shape)
	#img = img[tf.newaxis, :]
	return img

def gilbert():
	""" gibt die gilbert katze als tensor """
	url = "https://raw.githubusercontent.com/bomelino/stupid/master/images/gilbert.jpg"
	img = tf.image.decode_image(requests.get(url).content, channels=3) #, name="jpeg_reader")
	img = tf.image.convert_image_dtype(img, tf.float32)
	return img

def img(name):
	""" gibt bild aus dem github folder /images"""
	url = "https://raw.githubusercontent.com/bomelino/stupid/master/images/"+name
	img = tf.image.decode_image(requests.get(url).content, channels=3) #, name="jpeg_reader")
	img = tf.image.convert_image_dtype(img, tf.float32)
	return img


def get_img(src):
   img = imageio.imread(src)
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   return img

def norm(t):
  """ normalisiere einen numpy zu werten zwischen 0 und 1 """
  return (t - t.min()) / (t.max()-t.min())

def batch(t): 
	""" füge batch dimension hinzu """
	return tf.expand_dims(t,axis=0)

def to_image(obj):
  """ allgemeine funktion zum anschauen von allen objekttypen (work in progress)
  gibt image (numpy arry),description zurück
  description sagt, was alles gemacht wurde um bild darzustellen
  """
  import logging 
  descr = ""
  if (tf.is_tensor(obj)):
    obj = obj.numpy()
    

  logger = logging.getLogger()
  old_level = logger.level
  logger.setLevel(100)
  if obj.shape:
    #print(f"Max {max(obj)}")
    if len(obj.shape) == 2: # grayscale image
      obj = norm(obj)

      descr += f"Grayscale Image, mean:{obj.mean()}, var:{obj.var()} \n" 
      if (obj.var() < 0.01):
        descr += f"Mean abgzogen {obj.mean()} \n"
        obj = obj - obj.mean()

      if (obj.mean() < 0.01):        
        i = 0
        while (obj.mean() < 0.1 and obj.shape[0] > 10):
          i += 1
          obj = skimage.measure.block_reduce(obj, (2,2), np.max)
        descr += f"Sehr dunkles Bild, maxpooling ({i} mal)"
      # in "rgb" umwandeln
      obj = np.stack((obj,)*3, axis=-1)
      return obj,descr
    elif len(obj.shape) == 3: # könnte ein bild sein 

      if obj.shape[0] == 3:
        obj = np.transpose(obj,(1,2,0))
        descr += "channel first \n"

      if obj.shape[2] == 3: # normales bild 
        obj = norm(obj)
        descr += f"Mean {obj.mean()}, Variance {obj.var()}\n"
        if (obj.var() < 0.1):
          obj = obj - obj.mean()
          descr += f"Mean abgezogen \n"

        if (obj.mean() < 0.1):
          i= 0
          while (obj.mean() < 0.1 and obj.shape[0] > 10):
            i += 1
            obj = skimage.measure.block_reduce(obj, (2,2,1), np.max)
          descr += f"Bild zu dunkel, maxpooling ({i} mal)"
        
        return obj,descr
      else : ## feature map 
        ## zeige ein paar davon   
        
        n = math.floor(math.sqrt(obj.shape[2]/3))
        n = min(n,8)
      
        f, axs = plt.subplots(n,n,figsize=(15,15))    
        descr += f"{obj.shape[2]} Feature Maps mit Shape {obj.shape[0:2]}"
        print(f'Zeige {n*n*3} Feature Maps via RGB:')
        
        for i in range(n*n):
            r = norm(obj[:,:,i*3])
            g = norm(obj[:,:,i*3+1])
            b = norm(obj[:,:,i*3+2])    
            axs.flat[i].set_title(f'{i*3} - {i*3+3}')
            axs.flat[i].imshow(np.moveaxis(np.array([r,g,b]), 0, 2)) # channels first -> channels last
            #axs.flat[i].imshow(r,cmap='gray')
            axs.flat[i].axis('off')
    elif len(obj.shape) == 4 and obj.shape[0] == 3 and obj.shape[0] == 3: # convolution kernel
      descr += f"Convolution Kernel {obj.shape}"
      obj = np.transpose(obj,(2,3,0,1))
      obj = np.reshape(obj,(obj.shape[0],-1,3))
      #obj = obj[:,:,:3]
      return to_image(obj)
    
    else:
      print("Tensor ",obj.shape)
      print(obj)
    
    logger.setLevel(old_level)

  else:
    return None, "Object of type "+str(type(obj))



def view(obj,verbose=False):
    result,descr = to_image(obj)
    #if result != None:
    if verbose: print(descr)
    plt.imshow(result)
    plt.show()


def tnorm(t):
	return (t - tf.math.reduce_min(t)) / (tf.math.reduce_max(t)-tf.math.reduce_min(t))
  