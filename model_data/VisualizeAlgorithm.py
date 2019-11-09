""" Visualize the algorithm by plotting all nodes representing the potential steps that can be taken and a solid
	black line which connects these steps together"""

import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tkinter import *

results = []
with open("AlgorithmProcess.csv", "r") as f:
	header = f.readline()
	for line in f:
		results.append(list(map(float,line.strip().split(",")[:-1])))



print()
minimum = np.array(results).min()
maximum = np.array(results).max()

def get_color(error):
	perc = ((error - minimum) / (maximum - minimum))
	mycolor = '#%02x%02x%02x' % (int(perc*255), int((1-perc)*255), 0)  # set your favourite rgb color
	return mycolor

master = Tk()

w = Canvas(master, width=1200, height=450)
w.pack()



last_point = (0,0)
next_point = (0,0)
for i,result in enumerate(results):
	for j,val in enumerate(result):
		outline_str = ""
		if min(result) == val:
			outline_str = "black"
			next_point = (i*45+20,j*25+20)
		if last_point != (0,0):
			w.tag_lower(w.create_line(*last_point,i*45+20,j*25+20, width=2))
		w.create_oval(i*45+10,j*25+10,i*45+30,j*25+30,fill=get_color(val),outline=outline_str,width=5.0)
	last_point = next_point

import canvasvg
canvasvg.saveall("algo.svg", w)

mainloop()
