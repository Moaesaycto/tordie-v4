from settings import *
from abc import ABC, abstractmethod
from typing import Union

class Shape(ABC):
    def __init__(self, width=DEFAULT_THICKNESS, stroke=DEFAULT_STROKE):
        self.width = width
        self.stroke = stroke
    
    @abstractmethod
    def draw(svg): pass

    @abstractmethod
    def translate(vector): pass

    @abstractmethod
    def rotate(angle, *args): pass

    @abstractmethod
    def scale(scale, *args): pass
    
    def set_stroke(self, stroke: str):
        self.stroke = stroke

    def set_width(self, width: Union[int, float]):
        self.width = width
    