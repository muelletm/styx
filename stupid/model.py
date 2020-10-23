
import tensorflow.keras as keras
from .layer import Reflect2D
from . import checkpoint

from . import constants

import os


def encoder_vgg(layer=4,path=None):
	""" gibt decoder des vggs ab layer zurück 
	(layer parameter noch nicht implementiert, layer ist )
	"""
	#@hack
	global _DATAPATH
	if(not path):
	 	path = os.path.join(constants.datapath,"wct/relu4_1")
	
	cp = checkpoint.load(path)
	enc = cp.names["vgg_encoder"]
	enc.pretty()
	#print("encoder node:")
	#enc.pretty() # hier kann man sehen, wie der node tree aussieht

	structure = ["preprocess","reflect", "conv1_1","relu","reflect","conv1_2","relu","pool","reflect","conv2_1",
				 "relu","reflect","conv2_2","relu","pool","reflect","conv3_1","relu","reflect",
				 "conv3_2","relu","reflect","conv3_3","relu","reflect","conv3_4","relu","pool",
				 "reflect","conv4_1","relu"] #,"reflect","conv4_2","relu","reflect","conv4_3",
				 #"relu","reflect","conv4_4","relu","pool","reflect","conv5_1","relu"] 

	encoder = keras.Sequential()

	for name in structure:
		if name == "reflect":
			encoder.add(Reflect2D())
		elif name == "relu":
			encoder.add(keras.layers.Activation("relu"))
		elif name == "pool":
			encoder.add(keras.layers.MaxPooling2D(padding='same'))
		else:
			encoder.add(enc.children[name].toConv2D())
	return encoder


def decoder_vgg(layer=4,path=None):
	global _DATAPATH
	if(not path):
	 	path = os.path.join(constants.datapath,"wct/relu4_1")
	cp = checkpoint.load(path)

	# ----------------- baue decoder

	dec = cp.names["decoder_model_relu4_1"]
	#print("decoder node:")
	#dec.pretty()

	dec = dec.filter(lambda x: not x.isLayer()) # das sind die nodes ohne "_1" am ende
	dec.pretty()
	# struktur aus originalcode
	structure = ["relu4_1_0","upsample","relu4_1_2","relu4_1_3","relu4_1_4","relu4_1_5","upsample",
				 "relu4_1_7","relu4_1_8","upsample","relu4_1_10","relu4_1_11" ]

	decoder = keras.Sequential()
	for layer in structure:
		if layer == "upsample":
			decoder.add(keras.layers.UpSampling2D())
		elif layer == "relu4_1_11": ## letzte layer hat keine activation wer weiß warum
			decoder.add(Reflect2D())
			decoder.add(dec.children[layer].children[layer].toConv2D(activation=None))
		else: 
			decoder.add(Reflect2D())
			decoder.add(dec.children[layer].children[layer].toConv2D(activation='relu'))


	decoder.build(input_shape=(None,None,None,512)) # braucht man nur für die summary
	return decoder