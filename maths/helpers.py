from shapes.point import Point
import numpy as np
from options import *
import sys
import os

def dist(point1, point2): return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def diff(fx, t): return (fx(t + DIFF_STEP) - fx(t))/DIFF_STEP

def stopPrint(func, *args, **kwargs):
    with open(os.devnull,"w") as devNull:
        original = sys.stdout
        sys.stdout = devNull
        func(*args, **kwargs)
        sys.stdout = original 