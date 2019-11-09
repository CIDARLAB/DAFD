""" Gradient Descent visualized for each parameter on a sliding bar"""

from colour import Color
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


headers = []
data_entries = []
experimental_outputs = []
desired_outputs = []
with open("InterResults.csv","r") as f:
	experimental_outputs = f.readline().strip().split(":")[1].split(",")
	desired_outputs = f.readline().strip().split(":")[1].split(",")
	headers = f.readline().strip().split(",")
	for line in f:
		data_entries.append(list(map(float,line.strip().split(","))))
	first_results = data_entries[0]
	last_results = data_entries[-1]
	data_entries = data_entries

print(headers)
print(len(data_entries))

color_list = list(Color("red").range_to(Color("green"),len(data_entries)))
red_green_cmap = LinearSegmentedColormap.from_list("color_map",[(0,1,0),(1,0,0)])
loss_list = [x[11] for x in data_entries]
max_loss = max(loss_list)
min_loss = min(loss_list)
norm_loss_list = [(x-min_loss)/(max_loss-min_loss) for x in loss_list]


for i,header in enumerate(headers):
	fig = plt.figure(header.title().replace("_", " "), figsize=[10,6])
	plt.scatter(range(len(data_entries)), [x[i] for x in data_entries], c=norm_loss_list, cmap=red_green_cmap, s=150, zorder=10)
	plt.plot(range(len(data_entries)), [x[i] for x in data_entries], "k--", zorder=1)
	fig.suptitle(header.title().replace("_"," "), fontsize=26)
	plt.xlabel("Iteration", fontsize=22)
	plt.ylabel("Value", fontsize=22)
	plt.grid()
	ax = fig.axes[0]
	ax.tick_params(labelsize=15)
	ax.ticklabel_format(useOffset=False)
	fig.savefig('results/'+header+'.svg', format='svg', dpi=1200)

plt.show()