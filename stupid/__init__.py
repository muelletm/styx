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
from ops import * # das müsste ok sein, weil operations sehr spezielle namen haben und sich da nichts in die quere kommt

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



def batch(t): 
	""" füge batch dimension hinzu """
	return tf.expand_dims(t,axis=0)

