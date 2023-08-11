"""                             
_/_/_/_/_/                          _/  _/                _/  _/        _/    
   _/      _/_/    _/  _/_/    _/_/_/        _/_/        _/  _/      _/  _/   
  _/    _/    _/  _/_/      _/    _/  _/  _/_/_/_/      _/_/_/_/    _/  _/    
 _/    _/    _/  _/        _/    _/  _/  _/                _/      _/  _/     
_/      _/_/    _/          _/_/_/  _/    _/_/_/          _/  _/    _/

CC BY-NC-SA: 2022-2023

Tordie 4.0 is an Origami pattern generator with high mathematical functionality
designed to cross the bridge from ideas into practical SVG files to use for scoring.

Developer: Moaesaycto (S.L.)

Title generated by: https://ascii.today/
"""

from svg.svg import Diagram
from shapes import RegularPolygon, Polygon
from shapes import Point as P
from math import pi
from random import choice, randrange

if __name__ == '__main__':
    drawing = Diagram()
    random_hexagonm = Polygon([P(1, -1), P(1, 1), P(-1, 1), P(-1, -1)], m=[1, -1], show_points=True, m_center="relative")
    for _ in range(10): random_hexagonm = Polygon(random_hexagonm.points, m=[1, -1])
    drawing.draw(random_hexagonm)
    drawing.display()