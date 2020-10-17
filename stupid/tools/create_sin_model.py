"""Based on https://www.tensorflow.org/lite/guide/ops_custom#defining_the_kernel_in_the_tensorflow_lite_runtime."""
import argparse

import tensorflow as tf


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Converts H5 to TF lite.')
    parser.add_argument('tf_lite_model', help='TF lite model.')

    args = parser.parse_args()

    offset = tf.Variable(0.0)

    # Define a simple model which just contains a custom operator named `Sin`
    @tf.function
    def sin(x):
        return tf.sin(x + offset, name="Sin")


    xs = [-8, 0.5, 2, 2.2, 201]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([sin.get_concrete_function(xs)])
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    with open(args.tf_lite_model, 'wb') as f:
        f.write(tflite_model)
