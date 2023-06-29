"""                             
_/_/_/_/_/                          _/  _/                _/  _/        _/    
   _/      _/_/    _/  _/_/    _/_/_/        _/_/        _/  _/      _/  _/   
  _/    _/    _/  _/_/      _/    _/  _/  _/_/_/_/      _/_/_/_/    _/  _/    
 _/    _/    _/  _/        _/    _/  _/  _/                _/      _/  _/     
_/      _/_/    _/          _/_/_/  _/    _/_/_/          _/  _/    _/

CC BY-NC-SA: 2022/2023

Tordie 4.0 is an Origami pattern generator with high mathematical functionality
designed to cross the bridge from ideas into practical SVG files to use for scoring.

Developer: Moaesaycto (S.L.)

Title generated by: https://ascii.today/
"""

from init import init
from svg.svg import Diagram
from shapes.point import Point as P
from shapes.line import EuclideanLine as Line
from shapes.tessellation import MiuraOri

from math import pi, e, sin, cos, sqrt
from math import exp as e

from maths.helpers import *

def rad(p):
    el_rad = rad_poly_star_lambda(6, 2, 1)
    return p.x/e(p.x)*el_rad(p)

if __name__ == '__main__':
    init(short=True)
    drawing = Diagram()
    ori = MiuraOri(4, 24, v_height=2*pi, h_width=2*pi, parametric_verts=True, top=False, squash_factor=1, start=P(2, -pi))
    drawing.draw(ori.conformal(lambda t: exp(t, rad=rad)))
    drawing.display()