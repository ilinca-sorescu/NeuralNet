from random import uniform
import math

def printData(filename, num, maxval = 10):
    f = open(filename, "w")
    for i in range(num):
        r = uniform(0, maxval)
        h = uniform(0, maxval)
        f.write("{0} {1} {2}\n".format(r, h, math.pi*r*r*h))

printData("cylinder.train", 10000);
printData("cylinder.validate", 10);
