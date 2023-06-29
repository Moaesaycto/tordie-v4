from settings import *
from svgwrite.drawing import Drawing
from collections.abc import Iterable

import sys
import os

from tkinter import *

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from PIL import Image, ImageTk

from maths.helpers import stopPrint 

class Diagram:
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, name="result.svg"):
        self.width, self.height = width, height
        size = (f"{str(self.width + BOUNDS_FIX)}px", f"{str(self.height  + BOUNDS_FIX)}px")
        self.name = name
        self.drawing = Drawing('result.svg', size=size)
        self.drawing.add(self.drawing.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None, fill=BACKGROUND))

    def set_name(self, name):
        self.name = name

    def save(self, directory=""):
        if self.name[-4:] != ".svg": self.name = self.name + ".svg"
        self.drawing.saveas(directory + self.name)


    def draw(self, *args):
        for arg in args:
            if arg is None: continue
            if (isinstance(arg, Iterable)):
                for comp in arg: self.draw(comp)
            else: arg.draw(self)


    def display(self, name="temp.png"):
        self.save()
        if name[-4:] != ".png": name += ".png"

        root = Tk()
        root.title("Tordie - Result")
        root.geometry(f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

        display_frame = Canvas(root, bg='gray', width=DISPLAY_WIDTH+100, height=DISPLAY_HEIGHT+100)

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        display_frame.grid(column=1, row=0, sticky="ew")

        drawing = svg2rlg("result.svg")
        renderPM.drawToFile(drawing, name, fmt="PNG")
        img = Image.open('temp.png')
        img = img.resize((DISPLAY_WIDTH - 100, DISPLAY_HEIGHT - 100), Image.ANTIALIAS)
        pimg = ImageTk.PhotoImage(img)
        size = img.size

        display_frame.create_image(DISPLAY_WIDTH/2, DISPLAY_HEIGHT/2,anchor='c',image=pimg)

        root.mainloop()
