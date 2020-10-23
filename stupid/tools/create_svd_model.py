"""Based on https://www.tensorflow.org/lite/guide/ops_custom#defining_the_kernel_in_the_tensorflow_lite_runtime."""
import argparse

import tensorflow as tf


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Converts H5 to TF lite.')
    parser.add_argument('tf_lite_model', help='TF lite model.')

    args = parser.parse_args()

    def svd(x):
        s, u, _ = tf.linalg.svd(x)
        return s, u


    def build_model(batch_size=1):
        inputs = tf.keras.layers.Input(name="input", shape=[None, None], batch_size=1)
        outputs = svd(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    model = build_model()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    with open(args.tf_lite_model, 'wb') as f:
        f.write(tflite_model)
