from options import *
from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self, **kwargs):
        self.width = kwargs.get("width", DEFAULT_RADIUS)
        self.stroke = kwargs.get("stroke", DEFAULT_STROKE)
    
    @abstractmethod
    def draw(svg): pass