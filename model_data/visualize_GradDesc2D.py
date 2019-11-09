""" Visualization of the gradient descent as a 2D graph of generation_rate vs size for each step in the optimization"""

from tkinter import *
from colour import Color
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib


headers = []
data_entries = []
experimental_outputs = []
desired_outputs = []
with open("InterResults.csv","r") as f:
	experimental_outputs = list(map(float,f.readline().strip().split(":")[1].split(",")))
	desired_outputs = list(map(float,f.readline().strip().split(":")[1].split(",")))
	headers = f.readline().strip().split(",")
	for line in f:
		data_entries.append(line.strip().split(","))

print(headers)
print(len(data_entries))

drop_size_index = headers.index("droplet_size")
gen_rate_index = headers.index("generation_rate")

color_list = list(Color("red").range_to(Color("green"),len(data_entries)))


red_green_cmap = LinearSegmentedColormap.from_list("color_map",[(0,1,0),(1,0,0)])
loss_list = [float(x[11]) for x in data_entries]
max_loss = max(loss_list)
min_loss = min(loss_list)
norm_loss_list = [(x-min_loss)/(max_loss-min_loss) for x in loss_list]


fig = plt.figure("Gradient Descent", figsize=[9,9])

last_point = (experimental_outputs[1],experimental_outputs[0])
for i,point in enumerate(data_entries):
	drop_size = float(point[drop_size_index])
	gen_rate = float(point[gen_rate_index])
	plt.plot((last_point[0],drop_size),(last_point[1],gen_rate),"k--", zorder=0)
	plt.plot(drop_size, gen_rate, "ro", c=red_green_cmap(norm_loss_list[i]), zorder=5)
	print(last_point)
	print(drop_size)
	print(i/len(data_entries))
	last_point = (drop_size,gen_rate)

if desired_outputs[0] == -1:
	plt.plot((desired_outputs[1], desired_outputs[1]),(500, 0),"g--")
	print(desired_outputs[1])
elif desired_outputs[1] == -1:
	plt.plot((500,0),(desired_outputs[0], desired_outputs[0]),"g--")
else:
	plt.plot(desired_outputs[1], desired_outputs[0], "b^", zorder=10)

plt.plot(experimental_outputs[1], experimental_outputs[0], "m^")


plt.grid()
ax = fig.axes[0]
max_range = max([ax.get_xlim()[1]-ax.get_xlim()[0],ax.get_ylim()[1]-ax.get_ylim()[0]])
ax.set_xlim(((ax.get_xlim()[0] + ax.get_xlim()[1]) / 2) - max_range/2,((ax.get_xlim()[0] + ax.get_xlim()[1]) / 2) + max_range/2)
ax.set_ylim(((ax.get_ylim()[0] + ax.get_ylim()[1]) / 2) - max_range/2,((ax.get_ylim()[0] + ax.get_ylim()[1]) / 2) + max_range/2)
fig.savefig('results/grad_desc.svg', format='svg', dpi=1200)

ax = fig.axes[0]
ax.set_xlim(desired_outputs[1]-1, desired_outputs[1]+1)
ax.set_ylim(desired_outputs[0]-1, desired_outputs[0]+1)
fig.savefig('results/grad_desc_zoomedin.svg', format='svg', dpi=1200)

plt.show()
