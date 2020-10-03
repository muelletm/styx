"""
module documentation
"""
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

	g["tf"] = tf 
	g["display"] = display 
	g["plt"] = plt 
	g["mpl"] = mpl 
	g["numpy"] = np 
	g["np"] = np
	g["functools"] = functools 
	g["tfds"] = tfds

	print("verfügbare Variablen: tf, display, plt, np/numpy, functools, tfds")
