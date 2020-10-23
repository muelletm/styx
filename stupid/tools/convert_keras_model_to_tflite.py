import argparse

import tensorflow as tf
import stupid

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Converts H5 to TF lite.')
    parser.add_argument('keras_model', help='Keras model to load in H5 format.')
    parser.add_argument('tf_lite_model', help='TF lite model.')

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.keras_model, custom_objects={"Reflect2D": stupid.layer.Reflect2D})
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.allow_custom_ops = False
    tflite_model = converter.convert()

    with open(args.tf_lite_model, 'wb') as f:
        f.write(tflite_model)


