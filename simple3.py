
import stupid

g = stupid.graph.load("./graph2.pbtxt")

r = stupid.graph.io(g)
print("inputs:",r[0])
print("outputs:",r[1])

stupid.graph.tensorboard("graph2.pbtxt")