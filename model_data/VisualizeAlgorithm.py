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

w = Canvas(master, width=800, height=450)
w.pack()



for i,result in enumerate(results):
	for j,val in enumerate(result):
		outline_str = "black" if min(result)==val else ""
		w.create_oval(i*25+10,j*25+10,i*25+30,j*25+30,fill=get_color(val),outline=outline_str,width=5.0)

mainloop()
