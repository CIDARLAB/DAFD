"""A graphical interface to our interpolation modeller"""

from InterModel import InterModel
import tkinter
from tkinter import ttk
import tkinter.messagebox

class DAFD_GUI:
	"""A class that produces a windowed interface for DAFD"""

	def __init__(self):
		"""Initialize the GUI components"""
		self.root = tkinter.Tk()
		self.root.title("DAFD")


		#Attach the interpolation model to the GUI
		self.it = InterModel()

		ranges_dict = self.it.ranges_dict

		#Pack all input constraint elements together
		inputs_frame = tkinter.Frame(self.root)
		inputs_frame.pack(side="top")

		inputs_header = tkinter.Label(inputs_frame)
		inputs_header.pack(side="top")
		inputs_header["text"] = "Constraints"
		inputs_header.config(font=("Times", 20))

		orifice_size_frame = tkinter.Frame(inputs_frame)
		orifice_size_frame.pack(side="top")
		orifice_size_label = tkinter.Label(orifice_size_frame,width=40,anchor="e")
		orifice_size_label.pack(side="left")
		orifice_size_label["text"]="Orifice Size (µm) " + "(" + str(round(ranges_dict["orifice_size"][0],2)) + "-" + str(round(ranges_dict["orifice_size"][1],2)) + ")" + ": "
		self.orifice_size_entry = tkinter.Entry(orifice_size_frame)
		self.orifice_size_entry.pack(side="left")

		aspect_ratio_frame = tkinter.Frame(inputs_frame)
		aspect_ratio_frame.pack(side="top")
		aspect_ratio_label = tkinter.Label(aspect_ratio_frame,width=40,anchor="e")
		aspect_ratio_label.pack(side="left")
		aspect_ratio_label["text"]="Aspect ratio " + "(" + str(round(ranges_dict["aspect_ratio"][0],2)) + "-" + str(round(ranges_dict["aspect_ratio"][1],2)) + ")" + ": "
		self.aspect_ratio_entry = tkinter.Entry(aspect_ratio_frame)
		self.aspect_ratio_entry.pack(side="left")

		width_ratio_frame = tkinter.Frame(inputs_frame)
		width_ratio_frame.pack(side="top")
		width_ratio_label = tkinter.Label(width_ratio_frame,width=40,anchor="e")
		width_ratio_label.pack(side="left")
		width_ratio_label["text"]="Width Ratio " + "(" + str(round(ranges_dict["width_ratio"][0],2)) + "-" + str(round(ranges_dict["width_ratio"][1],2)) + ")" + ": "
		self.width_ratio_entry = tkinter.Entry(width_ratio_frame)
		self.width_ratio_entry.pack(side="left")

		normalized_orifice_length_frame = tkinter.Frame(inputs_frame)
		normalized_orifice_length_frame.pack(side="top")
		normalized_orifice_length_label = tkinter.Label(normalized_orifice_length_frame,width=40,anchor="e")
		normalized_orifice_length_label.pack(side="left")
		normalized_orifice_length_label["text"]="Normalized Orifice Length " + "(" + str(round(ranges_dict["normalized_orifice_length"][0],2)) + "-" + str(round(ranges_dict["normalized_orifice_length"][1],2)) + ")" + ": "
		self.normalized_orifice_length_entry = tkinter.Entry(normalized_orifice_length_frame)
		self.normalized_orifice_length_entry.pack(side="left")

		normalized_oil_input_width_frame = tkinter.Frame(inputs_frame)
		normalized_oil_input_width_frame.pack(side="top")
		normalized_oil_input_width_label = tkinter.Label(normalized_oil_input_width_frame,width=40,anchor="e")
		normalized_oil_input_width_label.pack(side="left")
		normalized_oil_input_width_label["text"]="Normalized Oil Input Width " + "(" + str(round(ranges_dict["normalized_oil_input_width"][0],2)) + "-" + str(round(ranges_dict["normalized_oil_input_width"][1],2)) + ")" + ": "
		self.normalized_oil_input_width_entry = tkinter.Entry(normalized_oil_input_width_frame)
		self.normalized_oil_input_width_entry.pack(side="left")

		normalized_water_input_width_frame = tkinter.Frame(inputs_frame)
		normalized_water_input_width_frame.pack(side="top")
		normalized_water_input_width_label = tkinter.Label(normalized_water_input_width_frame,width=40,anchor="e")
		normalized_water_input_width_label.pack(side="left")
		normalized_water_input_width_label["text"]="Normalized Water Input Width " + "(" + str(round(ranges_dict["normalized_water_input_width"][0],2)) + "-" + str(round(ranges_dict["normalized_water_input_width"][1],2)) + ")" + ": "
		self.normalized_water_input_width_entry = tkinter.Entry(normalized_water_input_width_frame)
		self.normalized_water_input_width_entry.pack(side="left")

		capillary_number_frame = tkinter.Frame(inputs_frame)
		capillary_number_frame.pack(side="top")
		capillary_number_label = tkinter.Label(capillary_number_frame,width=40,anchor="e")
		capillary_number_label.pack(side="left")
		capillary_number_label["text"]="Capillary Number " + "(" + str(round(ranges_dict["capillary_number"][0],2)) + "-" + str(round(ranges_dict["capillary_number"][1],2)) + ")" + ": "
		self.capillary_number_entry = tkinter.Entry(capillary_number_frame)
		self.capillary_number_entry.pack(side="left")

		flow_rate_ratio_frame = tkinter.Frame(inputs_frame)
		flow_rate_ratio_frame.pack(side="top")
		flow_rate_ratio_label = tkinter.Label(flow_rate_ratio_frame,width=40,anchor="e")
		flow_rate_ratio_label.pack(side="left")
		flow_rate_ratio_label["text"]="Flow Rate Ratio " + "(" + str(round(ranges_dict["flow_rate_ratio"][0],2)) + "-" + str(round(ranges_dict["flow_rate_ratio"][1],2)) + ")" + ": "
		self.flow_rate_ratio_entry = tkinter.Entry(flow_rate_ratio_frame)
		self.flow_rate_ratio_entry.pack(side="left")



		#Pack the desired output elements together
		outputs_frame = tkinter.Frame(self.root,pady=20)
		outputs_frame.pack(side="top")
		
		outputs_header = tkinter.Label(outputs_frame)
		outputs_header.pack(side="top")
		outputs_header["text"] = "Desired Values"
		outputs_header.config(font=("Times", 20))

		generation_rate_frame = tkinter.Frame(outputs_frame)
		generation_rate_frame.pack(side="top")
		generation_rate_label = tkinter.Label(generation_rate_frame,width=40,anchor="e")
		generation_rate_label.pack(side="left")
		generation_rate_label["text"]="Generation Rate (Hz): "
		self.generation_rate_entry = tkinter.Entry(generation_rate_frame)
		self.generation_rate_entry.pack(side="left")

		size_frame = tkinter.Frame(outputs_frame)
		size_frame.pack(side="top")
		size_label = tkinter.Label(size_frame,width=40,anchor="e")
		size_label.pack(side="left")
		size_label["text"]="Droplet Diameter (µm): "
		self.size_entry = tkinter.Entry(size_frame)
		self.size_entry.pack(side="left")



		#Pack the results together
		results_frame = tkinter.Frame(self.root,pady=20)
		results_frame.pack(side="top")
		submit_button = ttk.Button(results_frame, text='Run DAFD',command = self.runInterp)
		submit_button.pack(side="top")
		self.results_label = tkinter.Label(results_frame)
		self.results_label.pack(side="top")

		
		#Start everything
		self.root.mainloop()


	def runInterp(self):
		"""Predict an input set based on given constraints and desired outputs"""

		#Entry elements we need to collect from
		entries_list = [self.orifice_size_entry.get(),
				self.aspect_ratio_entry.get(),
				self.width_ratio_entry.get(),
				self.normalized_orifice_length_entry.get(),
				self.normalized_oil_input_width_entry.get(),
				self.normalized_water_input_width_entry.get(),
				self.capillary_number_entry.get(),
				self.flow_rate_ratio_entry.get()]

		#The keys to the returned values from our InterModel
		input_headers = ["orifice_size",
				"aspect_ratio",
				"width_ratio",
				"normalized_orifice_length",
				"normalized_oil_input_width",
				"normalized_water_input_width",
				"capillary_number",
				"flow_rate_ratio"]

		#Nice display to give to user
		input_headers_clean = ["Orifice Size",
				"Aspect Ratio",
				"Width Ratio",
				"Normalized Orifice Length",
				"Normalized Oil Input Width",
				"Normalized Water Input Width",
				"Capillary Number",
				"Flow Rate Ratio"]

		#Get all of our constraints
		constraints = {}
		for i in range(len(input_headers)):
			if(entries_list[i] != ""):
				#The constraint can either be a single value or a range
				if "-" in entries_list[i]:
					#If it is a single value x, the range is x-x
					pair = entries_list[i].split("-")
					wanted_constraint = (float(pair[0]),float(pair[1]))
				else:
					#If it is a range x to y, the range is x-y
					wanted_constraint = (float(entries_list[i]),float(entries_list[i]))

				if wanted_constraint[0] < self.it.ranges_dict[input_headers[i]][0]:
					tkinter.messagebox.showwarning("Out of range constraint",input_headers_clean[i] + " was too low. Constraint ignored")
				elif wanted_constraint[1] > self.it.ranges_dict[input_headers[i]][0]:
					tkinter.messagebox.showwarning("Out of range constraint",input_headers_clean[i] + " was too high. Constraint ignored")
				else:
					constraints[input_headers[i]] = wanted_constraint

		# Get the desired outputs
		# Note one can be left blank, in which case the interpolation model will simply operate on the other value's model
		desired_vals = {}
		if(self.size_entry.get()!=""):
			desired_vals["droplet_size"] = float(self.size_entry.get())
		if(self.generation_rate_entry.get()!=""):
			desired_vals["generation_rate"] = float(self.generation_rate_entry.get())

		#Return and display the results
		results = self.it.interpolate(desired_vals,constraints)
		self.results_label["text"] = "\n".join([str(input_headers_clean[x]) + " : " + str(results[input_headers[x]]) for x in range(len(input_headers))])



#Executed when script is called from console
DAFD_GUI()
