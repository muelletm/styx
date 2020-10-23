import tensorflow as tf 
import glom
import pprint 
pp = pprint.PrettyPrinter(indent=4,compact=True).pprint
import copy
from tensorflow.keras.layers import Conv2D, UpSampling2D
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import scipy.misc

from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Activation, Lambda, MaxPooling2D, Layer
from keras.engine import InputSpec

from stupid.layer import *
from stupid import *
import stupid.checkpoint as checkpoint

file = "relu4_1/model.ckpt-15003"

# lade die variablen einer checkpoint datei
# rückgabe ist ein node
cp = checkpoint.load(file)

# ----------------- baue decoder

dec = cp.names["decoder_model_relu4_1"]
#print("decoder node:")
#dec.pretty()

dec = dec.filter(lambda x: not x.isLayer()) # das sind die nodes ohne "_1" am ende

# struktur aus originalcode
structure = ["relu4_1_0","upsample","relu4_1_2","relu4_1_3","relu4_1_4","relu4_1_5","upsample",
             "relu4_1_7","relu4_1_8","upsample","relu4_1_10","relu4_1_11" ]

decoder = keras.Sequential()
for layer in structure:
  if layer == "upsample":
      decoder.add(UpSampling2D())
  elif layer == "relu4_1_11": ## letzte layer hat keine activation wer weiß warum
      decoder.add(Reflect2D())
      decoder.add(dec.children[layer].children[layer].toConv2D(activation=None))
  else: 
      decoder.add(Reflect2D())
      decoder.add(dec.children[layer].children[layer].toConv2D(activation='relu'))


decoder.build(input_shape=(None,None,None,512)) # braucht man nur für die summary
print(decoder.summary())

## --------- baue vgg encoder

enc = cp.names["vgg_encoder"]
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
    encoder.add(Activation("relu"))
  elif name == "pool":
    encoder.add(MaxPooling2D(padding='same'))
  else:
    encoder.add(enc.children[name].toConv2D())


content = gilbert()
style = load("./images/style5.png")
fm1 = encoder(batch(content))
fm2 = encoder(batch(style))

# feature maps vom transformierten bild
fm = wct(fm1,fm2)

stylized = decoder.predict(fm) # funktioniert



## als keras model speichern

icontent = keras.Input(shape=(None,None,3),name="content")
istyle = keras.Input(shape=(None,None,3),name="style")

fm1 = encoder(icontent)
fm2 = encoder(istyle)

# feature maps vom transformierten bild
#fm = wct(fm1,fm2)
layer = tf.keras.layers.Lambda(lambda x:wct(x[0],x[1]))
fm = layer([fm1,fm2])
stylized = decoder(fm)

model = keras.Model(inputs=[icontent,istyle], outputs=[stylized], name="style_transfer")

result = model({"content":batch(content),"style":batch(style)})

#tflite_convert.exe --keras_model_file relu5.h5 --output_file relu5.tflite
#ValueError: Unknown layer: Reflect2D

#model.save("relu5.h5")

#model = tf.keras.models.load_model("relu5.h5",custom_objects={'Reflect2D': Reflect2D})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
tflite_model = converter.convert() # ValueError: None is only supported in the 1st dimension. Tensor 'content' has invalid shape '[None, None, None, 3]'.
open("relu5.tflite","wb").write(tflite_model)

