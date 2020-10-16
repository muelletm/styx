import stupid
#stupid.imports(globals())

import tensorflow.compat.v1 as tf

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution() # sonst gibt es keinen graph

folder = "relu4_1"


# standard objekt inspect
def inspect(obj):
	for key in dir(obj):
		if(not key.startswith("_")):
			value = getattr(obj,key) 
			print(f"{key}: {type(value)}")
			if(value.__doc__ != None):
				print("\t"+value.__doc__.split("\n")[0])


from google.protobuf import text_format

# lade pbtxt datei (untested)
def load_pb(file):
	f = open(file, "r")
	graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())
	# Import the graph protobuf into our new graph.
	g = tf.Graph()
	with g.as_default():
		tf.import_graph_def(graph_def=graph_protobuf, name="")
	return g

# lade .meta datei (untested)
def load_meta(file):
	# braucht man den saver?
	g = tf.Graph()
	with g.as_default():
		tf.train.import_meta_graph(file)
	return g


g = load_meta('relu4_1/model.ckpt-15003.meta')

# tensorflow graph
#g = tf.get_default_graph()


# g.get_tensor_by_name
#inspect(g)

# liste von variable objekten
variables = g.get_collection("variables")
print(variables)


d = g.as_graph_def()


