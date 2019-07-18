from tkinter import *
import colour
from colour import Color


headers = []
data_entries = []
experimental_outputs = []
desired_outputs = []
with open("InterResults.csv","r") as f:
	experimental_outputs = f.readline().strip().split(":")[1].split(",")
	desired_outputs = f.readline().strip().split(":")[1].split(",")
	headers = f.readline().strip().split(",")
	for line in f:
		data_entries.append(line.strip().split(","))
	data_entries = data_entries[1:-1]

print(headers)
print(len(data_entries))

color_list = list(Color("red").range_to(Color("green"),len(data_entries)))


tk = Tk()

frame = Frame(tk)
canvas = Canvas(frame, width=500, height=1000)

frame.pack(fill=BOTH, expand=1)
canvas.pack(fill=BOTH, expand=1)

slider_height = 30
slider_width = 10
sliders = []
sliders_y_pos = []
slider_texts = []

slider_bg_offset = 100
slider_bg_width = 300

last_y = 0
for header in headers:
	header_text = header.replace("_", " ").title()
	slider_text = canvas.create_text(slider_bg_offset + (slider_bg_width/2.0),last_y+30, font="Times 10", text=header_text)
	canvas.create_rectangle(slider_bg_offset,last_y+55,slider_bg_offset+slider_bg_width,last_y+65, outline="", fill="#CECECE")
	slider = canvas.create_rectangle(10,last_y+45,10+slider_width,last_y+45+slider_height, outline="", fill="red")
	sliders.append(slider)
	slider_texts.append(slider_text)
	sliders_y_pos.append(last_y+45)
	last_y = last_y+80

min_max_dict = {}
for i,header in enumerate(headers):
	min_val = float(data_entries[0][i])
	max_val = float(data_entries[0][i])
	for entry in data_entries:
		entry = float(entry[i])
		if entry < min_val:
			min_val = entry
		if entry > max_val:
			max_val = entry
	if min_val == max_val:
		min_val -= 1
		max_val += 1
	min_max_dict[header] = (min_val,max_val)

experimental_generation_rate = float(experimental_outputs[0])
minimum = min_max_dict["generation_rate"][0]
maximum = min_max_dict["generation_rate"][1]
normal = (experimental_generation_rate-minimum)/(maximum-minimum)
point_pos_x = slider_bg_offset + slider_bg_width*normal
point_pos_y = sliders_y_pos[9]
canvas.create_oval(point_pos_x,point_pos_y+10,point_pos_x+10,point_pos_y+20,outline="",fill="#FF00DC")

experimental_droplet_size = float(experimental_outputs[1])
minimum = min_max_dict["droplet_size"][0]
maximum = min_max_dict["droplet_size"][1]
normal = (experimental_droplet_size-minimum)/(maximum-minimum)
point_pos_x = slider_bg_offset + slider_bg_width*normal
point_pos_y = sliders_y_pos[10]
canvas.create_oval(point_pos_x,point_pos_y+10,point_pos_x+10,point_pos_y+20,outline="",fill="#FF00DC")

desired_generation_rate = float(desired_outputs[0])
if desired_generation_rate != -1:
	minimum = min_max_dict["generation_rate"][0]
	maximum = min_max_dict["generation_rate"][1]
	normal = (desired_generation_rate-minimum)/(maximum-minimum)
	point_pos_x = slider_bg_offset + slider_bg_width*normal
	point_pos_y = sliders_y_pos[9]
	canvas.create_oval(point_pos_x,point_pos_y+10,point_pos_x+10,point_pos_y+20,outline="",fill="#3FFF48")

desired_droplet_size = float(desired_outputs[1])
if desired_droplet_size != -1:
	minimum = min_max_dict["droplet_size"][0]
	maximum = min_max_dict["droplet_size"][1]
	normal = (desired_droplet_size-minimum)/(maximum-minimum)
	point_pos_x = slider_bg_offset + slider_bg_width*normal
	point_pos_y = sliders_y_pos[10]
	canvas.create_oval(point_pos_x,point_pos_y+10,point_pos_x+10,point_pos_y+20,outline="",fill="#3FFF48")


def next_frame(current_index):
	for i, header in enumerate(headers):
		minimum = min_max_dict[header][0]
		maximum = min_max_dict[header][1]
		header_text = header.replace("_", " ").title()
		value = data_entries[current_index][i]
		normal = (float(value)-minimum)/(maximum-minimum)
		canvas.itemconfig(slider_texts[i],text=header_text+": "+value)
		slider_new_pos = slider_bg_offset + slider_bg_width*normal
		canvas.coords(sliders[i], (slider_new_pos,sliders_y_pos[i],slider_new_pos+slider_width,sliders_y_pos[i]+slider_height))
		canvas.itemconfig(sliders[i],fill=str(color_list[current_index]))
		if current_index == 0 and header != "generation_rate" and header != "droplet_size":
			canvas.create_oval(slider_new_pos,sliders_y_pos[i]+10,slider_new_pos+10,sliders_y_pos[i]+20,outline="",fill="#FF00DC")
	#canvas.coords(ball, (10,40,300,60))  # change coordinates
	#canvas.coords(ball_text, (155,10))  # change coordinates
	#canvas.itemconfig(ball, fill="blue")  # change color
	canvas.update()

	if current_index < len(data_entries)-1:
		canvas.after(200, next_frame, current_index+1)

next_frame(0)

tk.mainloop()

