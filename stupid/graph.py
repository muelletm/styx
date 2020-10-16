""" funktionen um tensorflow graphen anzuschauen  / analysieren """

import tensorflow as tf
from google.protobuf import text_format

from tensorflow.compat.v1 import GraphDef
from tensorflow.python.summary import summary
from tensorflow.python.client import session

def save(g = None,file="graph.pb"):
	g = g or tf.compat.v1.get_default_graph()
	tf.io.write_graph(graph_or_graph_def=g,logdir=".",name=file)

# lade pbtxt, pb oder meta datei 
def load(file):
	if(file.endswith(".pbtxt")):
		f = open(file, "r")
		protobuf = text_format.Parse(f.read(), GraphDef())
		# Import the graph protobuf into our new graph.
		g = tf.Graph()
		with g.as_default():
			tf.import_graph_def(graph_def=protobuf, name="")
		return g
	elif (file.endswith(".pb")):
		f = open(file,"rb")
		graph_def = GraphDef()
		graph_def.ParseFromString(f.read())
		g = tf.Graph()
		with g.as_default():
			tf.import_graph_def(graph_def, name='')
		return g
	elif file.endswith(".meta"):
		g = tf.Graph()
		with g.as_default():
			tf.train.import_meta_graph(file)
		return g

def io(graph):
	""" finde input und output nodes (funktioniert nicht)
	notizen:
	sachen die keine outputs sind (normalerweise):
	VarIsInitializedOp, ReadVariableOp, Identity, Const, AssignVariableOp (?)
	"""
	ops = graph.get_operations()
	outputs_set = set(ops)
	inputs = []
	for op in ops:
		if len(op.inputs) == 0 and op.type != 'Const':
			inputs.append(op)
		else:
			for input_tensor in op.inputs:
				if input_tensor.op in outputs_set:
					outputs_set.remove(input_tensor.op)
	outputs = list(outputs_set)
	return (inputs, outputs)


def tensorboard(file,logdir="./log"):
	""" rufe danach in der shell 'tensorboard --log-dir log' auf """
	g = load(file)
	with session.Session(graph=g) as sess:
		w = summary.FileWriter(logdir)
		w.add_graph(sess.graph)