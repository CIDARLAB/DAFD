from InterModel import InterModel
import tkinter
from tkinter import ttk

class DAFD_GUI:
	def __init__(self):
		self.root = tkinter.Tk()


		inputs_frame = tkinter.Frame(self.root)
		inputs_frame.pack(side="top")

		inputs_header = tkinter.Label(inputs_frame)
		inputs_header.pack(side="top")
		inputs_header["text"] = "Constraints"
		inputs_header.config(font=("Times", 20))

		orifice_size_frame = tkinter.Frame(inputs_frame)
		orifice_size_frame.pack(side="top")
		orifice_size_label = tkinter.Label(orifice_size_frame,width=20,anchor="e")
		orifice_size_label.pack(side="left")
		orifice_size_label["text"]="Orifice Size: "
		self.orifice_size_entry = tkinter.Entry(orifice_size_frame)
		self.orifice_size_entry.pack(side="left")

		aspect_ratio_frame = tkinter.Frame(inputs_frame)
		aspect_ratio_frame.pack(side="top")
		aspect_ratio_label = tkinter.Label(aspect_ratio_frame,width=20,anchor="e")
		aspect_ratio_label.pack(side="left")
		aspect_ratio_label["text"]="Aspect ratio: "
		self.aspect_ratio_entry = tkinter.Entry(aspect_ratio_frame)
		self.aspect_ratio_entry.pack(side="left")

		width_ratio_frame = tkinter.Frame(inputs_frame)
		width_ratio_frame.pack(side="top")
		width_ratio_label = tkinter.Label(width_ratio_frame,width=20,anchor="e")
		width_ratio_label.pack(side="left")
		width_ratio_label["text"]="Width Ratio: "
		self.width_ratio_entry = tkinter.Entry(width_ratio_frame)
		self.width_ratio_entry.pack(side="left")

		normalized_orifice_length_frame = tkinter.Frame(inputs_frame)
		normalized_orifice_length_frame.pack(side="top")
		normalized_orifice_length_label = tkinter.Label(normalized_orifice_length_frame,width=20,anchor="e")
		normalized_orifice_length_label.pack(side="left")
		normalized_orifice_length_label["text"]="Normalized Orifice Length: "
		self.normalized_orifice_length_entry = tkinter.Entry(normalized_orifice_length_frame)
		self.normalized_orifice_length_entry.pack(side="left")

		normalized_oil_input_width_frame = tkinter.Frame(inputs_frame)
		normalized_oil_input_width_frame.pack(side="top")
		normalized_oil_input_width_label = tkinter.Label(normalized_oil_input_width_frame,width=20,anchor="e")
		normalized_oil_input_width_label.pack(side="left")
		normalized_oil_input_width_label["text"]="Normalized Oil Input Width: "
		self.normalized_oil_input_width_entry = tkinter.Entry(normalized_oil_input_width_frame)
		self.normalized_oil_input_width_entry.pack(side="left")

		normalized_water_input_width_frame = tkinter.Frame(inputs_frame)
		normalized_water_input_width_frame.pack(side="top")
		normalized_water_input_width_label = tkinter.Label(normalized_water_input_width_frame,width=20,anchor="e")
		normalized_water_input_width_label.pack(side="left")
		normalized_water_input_width_label["text"]="Normalized Water Input Width: "
		self.normalized_water_input_width_entry = tkinter.Entry(normalized_water_input_width_frame)
		self.normalized_water_input_width_entry.pack(side="left")

		capillary_number_frame = tkinter.Frame(inputs_frame)
		capillary_number_frame.pack(side="top")
		capillary_number_label = tkinter.Label(capillary_number_frame,width=20,anchor="e")
		capillary_number_label.pack(side="left")
		capillary_number_label["text"]="Capillary Number: "
		self.capillary_number_entry = tkinter.Entry(capillary_number_frame)
		self.capillary_number_entry.pack(side="left")

		flow_rate_ratio_frame = tkinter.Frame(inputs_frame)
		flow_rate_ratio_frame.pack(side="top")
		flow_rate_ratio_label = tkinter.Label(flow_rate_ratio_frame,width=20,anchor="e")
		flow_rate_ratio_label.pack(side="left")
		flow_rate_ratio_label["text"]="Flow Rate Ratio: "
		self.flow_rate_ratio_entry = tkinter.Entry(flow_rate_ratio_frame)
		self.flow_rate_ratio_entry.pack(side="left")




		outputs_frame = tkinter.Frame(self.root,pady=20)
		outputs_frame.pack(side="top")
		
		outputs_header = tkinter.Label(outputs_frame)
		outputs_header.pack(side="top")
		outputs_header["text"] = "Desired Values"
		outputs_header.config(font=("Times", 20))

		generation_rate_frame = tkinter.Frame(outputs_frame)
		generation_rate_frame.pack(side="top")
		generation_rate_label = tkinter.Label(generation_rate_frame,width=20,anchor="e")
		generation_rate_label.pack(side="left")
		generation_rate_label["text"]="Generation Rate: "
		self.generation_rate_entry = tkinter.Entry(generation_rate_frame)
		self.generation_rate_entry.pack(side="left")

		size_frame = tkinter.Frame(outputs_frame)
		size_frame.pack(side="top")
		size_label = tkinter.Label(size_frame,width=20,anchor="e")
		size_label.pack(side="left")
		size_label["text"]="Droplet Size: "
		self.size_entry = tkinter.Entry(size_frame)
		self.size_entry.pack(side="left")



		results_frame = tkinter.Frame(self.root,pady=20)
		results_frame.pack(side="top")
		submit_button = ttk.Button(results_frame, text='Run Interpolation',command = self.runInterp)
		submit_button.pack(side="top")
		self.results_label = tkinter.Label(results_frame)
		self.results_label.pack(side="top")

		self.it = InterModel()
		
		self.root.mainloop()


	def runInterp(self):
		entries_list = [self.orifice_size_entry.get(),
				self.aspect_ratio_entry.get(),
				self.width_ratio_entry.get(),
				self.normalized_orifice_length_entry.get(),
				self.normalized_oil_input_width_entry.get(),
				self.normalized_water_input_width_entry.get(),
				self.capillary_number_entry.get(),
				self.flow_rate_ratio_entry.get()]

		input_headers = ["orifice_size",
				"aspect_ratio",
				"width_ratio",
				"normalized_orifice_length",
				"normalized_oil_input_width",
				"normalized_water_input_width",
				"capillary_number",
				"flow_rate_ratio"]

		constraints = {}
		for i in range(len(input_headers)):
			if(entries_list[i] != ""):
				if "-" in entries_list[i]:
					pair = entries_list[i].split("-")
					constraints[input_headers[i]] = (float(pair[0]),float(pair[1]))
				else:
					constraints[input_headers[i]] = (float(entries_list[i]),float(entries_list[i]))

		desired_vals = {}
		if(self.size_entry.get()!=""):
			desired_vals["droplet_size"] = float(self.size_entry.get())
		if(self.generation_rate_entry.get()!=""):
			desired_vals["generation_rate"] = float(self.generation_rate_entry.get())

		results = self.it.interpolate(desired_vals,constraints)
		self.results_label["text"] = "\n".join([str(x) + " : " + str(results[x]) for x in results])

DAFD_GUI()
