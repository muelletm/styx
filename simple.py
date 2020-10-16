import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution


disable_eager_execution() # sonst gibt es keinen graph

# Let's explicitly create an empty graph: `g1`.
#
# Note, tensorflow has a default graph that can be used but we 
# explicitly create `g1` for clarity.
g1 = tf.Graph()

# `my_input_value` is a tensor-like object. 
# We provide a simple scalar and a numpy array as examples.

# my_input_value = np.random.multivariate_normal(
#     mean=(1,1),
#     cov=[[1,0], [0,1]],
#     size=10
# )

# simple scalar value
my_input_value = 2

# We want our operations to be placed on `g1` and not the default graph.
with g1.as_default():
    
    # Tensorflow operations usually take in a Tensor-like type, a data type, and a name.
    my_input = tf.constant(my_input_value, dtype=tf.float32, name="input")
    
    # These will be implicitly dtype `tf.float16`
    a = tf.square(my_input, name="A")
    b = tf.cos(a, name="B")
    c = tf.sin(a, name="C")   
    d = tf.add(b, c, name="D")
    e = tf.floor(b, name="E")
    f = tf.sqrt(d, name="F")

# We can write the graph as protobuf text file
tf.train.write_graph(graph_or_graph_def=g1, 
                     logdir='.', 
                     name='graph_protobuf.pbtxt')

# We can write out our graph to tensorboard for visualization
tf.summary.FileWriter("logs", g1).close()