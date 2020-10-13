"""
helper um mit checkpoints umzugehn 
"""
import tensorflow as tf
import glom
from tensorflow.keras.layers import Conv2D


def load(file):
	""" diese funktion von außen benutzen. gibt den root node zurück """
	return Node.fromFilename(file)

def set_path(obj,path,value):
	path = path.replace("/",".")+".value"
	glom.assign(obj, path, "v:"+str(value), missing=dict)
	return obj 

def to_obj(varlist):
	""" wandle die liste von tf.train.list_variables in ein dict um """
	result = {}
	for var in varlist:
		set_path(result,var[0],var[1])
	return result

"""
			# finde alle listen
			# tiefensuche
def find_list(tree,root):
	#print("find_list",tree,tree.id,tree.length())        
	if (tree.length() > 1 and not tree.hasChild("bias") and not tree.hasChild("Adam")):
		root.addList(tree)
		for key,value in tree.children.items():
			find_list(value,root)
	else:
		for key,value in tree.children.items():
			print(key,value)
			find_list(value,root)
"""

class Node:
	""" ein element einer baumstruktur. nicht dasselbe wie eine nodedef eines tensorflow graphen
	darstellung aller variablen in einer checkpoint datei.
	lädt die struktur nur aus den namen der variablen aus tf.train.list_variables(file)
	"""
	id = 0

	@classmethod
	def get_id(cls):
		Node.id += 1 
		return Node.id

	@classmethod 
	def fromFilename(cls,file):
		vars = tf.train.list_variables(file)
		obj = to_obj(vars)
		ckpt_reader = tf.train.load_checkpoint(file)
		t = Node.create(obj,"")
		t.reader = ckpt_reader  # root hat reader objekt
		return t

	@classmethod
	def create(cls,obj,name="",parent=None,root=None):
		""" erstelle Node von dict object """
		t = Node(name,parent,root)

		root = t.root

		for key,value in obj.items():      
			if (key == "value"): ### blatt
				t.value = value
				root.addLeaf(t)
			else:
				t.children[key] = Node.create(value,key,t,root)

		# root initialisieren
		if (t.root == t):   
			root.all_layers = []
			for l in t.leafs:
					if l.name == "bias":
						if (l.parent.hasChild("kernel")):
							root.all_layers.append(t.parent)
						# t.parent ist eine layer
						# füge t.parent.parent.layers hinzu
						if (l.parent.parent != None):
							l.parent.parent.layers.append(l.parent)
		return t

	def __init__(self,name="",parent=None,root=None):
		self.name = name 
		self.parent = parent
		self.root = root or self
		if (self.root == self):
			self.names = {} ## root dict
			self.leafs = []
			self.ids = {}
		
		self.id = Node.get_id()
		self.children = {}
		self.layers = []

		self.root.names[self.name] = self # alle schnell verfügbar, falls name eindeutig
		self.root.ids[self.id] = self

	def getTensor(self): 
		""" funktioniert nur, wenn root einen reader initialisiert hat """
		return self.root.reader.get_tensor(self.getPath()) 

	def getBias(self):
		""" für layer Nodes """
		return self.children["bias"].getTensor()
	def getWeights(self):
		""" für layer Nodes """
		return self.children["kernel"].getTensor()

	def hasChild(self,name):
		return name in self.children 
	def getPath(self):
			c = self
			path = ""
			while c != self.root:
				path = "/" + c.name + path
				c = c.parent
			return path[1:] # ersten slash weglassen
	def addLeaf(self,node):
		self.leafs.append(node)

	def length(self):
		return len(self.children)

	def isLayer(self):
		return self.hasChild("bias") and self.hasChild("kernel")
	def isList(self):
		return not self.isLayer() and len(self.layers) > 0
	def __repr__(self):
		if self.isLayer():
			return f"* {self.name}"
		elif self.isList():
			return f"[{self.length()}] {self.name}"
		else:
			return f"+ {self.name}" 



	def _pretty(self,indent = 0):
		""" drucke hübsch"""
		s = str(self) + "\n"
		for key,value in self.children.items():
			s += "   " * indent + value._pretty(indent+1)
		return s
	def pretty(self):
		s = self._pretty(1)
		print(s)
		return s

	def copy(self):
		t = Node(self.name,self.parent,self.root)
		t.children = self.children.copy()
		t.layers = self.layers
		t.id = self.id
		return t

	def filterSuffix(self,suff):
		""" lösche alle kinder die nicht dieses suffix haben"""
		return self.filter(lambda x: x.name.endswith(suff))
		
	def removeSuffix(self,suff):
		""" lösche das suffix aus den keys """
		return self.transformKeys(lambda x: x.name[:-len(suff)] if (x.name.endswith(suff)) else x.name)
		
	def transformKeys(self,f):
		""" gib einen node zurück, der alle keys k der kinder mit f(k) ersetzt hat """
		t = self.copy()
		t.children = {}
		for key,c in self.children.items():
			t.children[f(c)] = c
		return t
	def transform(self,f):
		""" gib einen node zurück, der alle kinder c mit f(c) ersetzt hat """
		t = self.copy()
		t.children = {}
		for key,c in self.children.items():
			t.children[key] = f(c)
		return t
	def filter(self, f):
		""" gib einen node zurück, der nur noch die kinder c hat, für die f(c) == True """
		t = self.copy()
		t.children = {}
		for key,c in self.children.items():
			if f(c):
				t.children[key] = c
		return t

	def sorted(self,pre):
		""" gibt liste von Nodes zurück, sortiert nach dem, was übrigbleibt, wenn man prefix löscht
		nimmt an, dass alle keys dasselbe format haben: prefix_number
		@todo: prefix selbst erkennen
		"""

		l = [ (int(x.name[len(pre):]),x) for x in self.children.copy().values() ]
		return [x[1] for x in sorted(l, key=lambda x: x[0])]

	def toConv2D(self,**args):
		""" konvertiere diesen node in eine keras conv2d layer.
		funktioniert nur, wenn self.isLayer() """
		bias = self.getBias()
		weights = self.getWeights()
		kernel_size = (weights.shape[0],weights.shape[1])
		filters = weights.shape[3]
		return Conv2D(filters, kernel_size, bias_initializer=tf.keras.initializers.Constant(bias),
										kernel_initializer=tf.keras.initializers.Constant(weights), **args)
		
 
