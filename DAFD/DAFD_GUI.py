"""A graphical interface to DAFD that does not require website hosting"""
from DAFD.helper_scripts.ModelHelper import ModelHelper
import tkinter
from PIL import ImageTk, Image
from tkinter import ttk
import tkinter.messagebox
from DAFD.bin.DAFD_Interface import DAFD_Interface

class DAFD_GUI:
	"""A class that produces a windowed interface for DAFD"""

	def __init__(self):
		"""Initialize the GUI components"""
		self.root = tkinter.Tk()
		self.root.title("DAFD")


		#Attach the interpolation model to the GUI
		self.di = DAFD_Interface()
		self.MH = ModelHelper.get_instance()  # type: ModelHelper

		# DAFD Logo
		img = Image.open(self.MH.resource_path("DAFD_logo.png"))
		img = ImageTk.PhotoImage(img.resize((int(0.1*img.size[0]), int(0.1*img.size[1])), Image.ANTIALIAS))
		panel = tkinter.Label(self.root, image=img)
		panel.pack(side="top", fill="both")
		panel.configure(background="white")
		self.root.configure(background="white")

		# Pack all input constraint elements together
		inputs_frame = tkinter.Frame(self.root)
		inputs_frame.pack(side="top")
		inputs_frame.configure(background="white")

		inputs_header = tkinter.Label(inputs_frame)
		inputs_header.pack(side="top")
		inputs_header["text"] = "Constraints"
		inputs_header.config(font=("Times", 20))
		inputs_header.configure(background="white")

		self.entries_dict = {}
		self.tol_var = 0

		for param_name in self.di.input_headers:
			param_frame = tkinter.Frame(inputs_frame)
			param_frame.pack(side="top")
			param_frame.configure(background="white")
			param_label = tkinter.Label(param_frame,width=40,anchor="e")
			param_label.pack(side="left")
			param_label["text"] = param_name + " (" + str(round(self.di.ranges_dict[param_name][0],2)) + "-" + str(round(self.di.ranges_dict[param_name][1],2)) + ") : "
			param_label.configure(background="white")
			param_entry = tkinter.Entry(param_frame)
			param_entry.pack(side="left")
			param_entry.configure(background="white")
			self.entries_dict[param_name] = param_entry

		regime_frame = tkinter.Frame(inputs_frame)
		regime_frame.pack(side="top")
		regime_frame.configure(background="white")
		regime_label = tkinter.Label(regime_frame,width=40,anchor="e")
		regime_label.pack(side="left")
		regime_label["text"] = "regime (1-2) : "
		regime_label.configure(background="white")
		regime_entry = tkinter.Entry(regime_frame)
		regime_entry.pack(side="left")
		regime_entry.configure(background="white")
		self.entries_dict["regime"] = regime_entry

		# Pack the desired output elements together
		outputs_frame = tkinter.Frame(self.root,pady=20)
		outputs_frame.pack(side="top")
		outputs_frame.configure(background="white")

		outputs_header = tkinter.Label(outputs_frame)
		outputs_header.pack(side="top")
		outputs_header["text"] = "Desired Values"
		outputs_header.config(font=("Times", 20))
		outputs_header.configure(background="white")

		for param_name in self.di.output_headers:
			param_frame = tkinter.Frame(outputs_frame)
			param_frame.pack(side="top")
			param_frame.configure(background="white")
			param_label = tkinter.Label(param_frame,width=40,anchor="e")
			param_label.pack(side="left")
			param_label["text"] = param_name + " (" + str(round(self.di.ranges_dict[param_name][0],2)) + "-" + str(round(self.di.ranges_dict[param_name][1],2)) + ") : "
			param_label.configure(background="white")
			param_entry = tkinter.Entry(param_frame)
			param_entry.pack(side="left")
			param_entry.configure(background="white")
			self.entries_dict[param_name] = param_entry

		# Pack Tolerance Study Together
		tolerance_frame = tkinter.Frame(self.root, pady=20)
		tolerance_frame.pack(side="top")
		tolerance_frame.configure(background="white")

		tolerance_header = tkinter.Label(tolerance_frame)
		tolerance_header.pack(side="top")
		tolerance_header["text"] = "Tolerance Options"
		tolerance_header.config(font=("Times", 20))
		tolerance_header.configure(background="white")

		tol_frame = tkinter.Frame(tolerance_frame)
		tol_frame.pack(side="top")
		tol_frame.configure(background="white")
		tol_label = tkinter.Label(tol_frame,width=40,anchor="e")
		tol_label.pack(side="left")
		tol_label["text"] = "Perform Tolerance Study? "
		tol_label.configure(background="white")
		check_var = tkinter.IntVar(value=0)
		tol_entry = tkinter.Checkbutton(tol_frame, variable=check_var)
		tol_entry.pack(side="left")
		tol_entry.configure(background="white")
		self.entries_dict["tolerance_test"] = tol_entry


		tol_frame = tkinter.Frame(tolerance_frame)
		tol_frame.pack(side="top")
		tol_frame.configure(background="white")
		tol_label = tkinter.Label(tol_frame,width=40,anchor="e")
		tol_label.pack(side="left")
		tol_label["text"] = "Tolerance (1-50, default 10) : "
		tol_label.configure(background="white")
		tol_entry = tkinter.Entry(tol_frame)
		tol_entry.pack(side="left")
		tol_entry.configure(background="white")
		self.entries_dict["tolerance"] = tol_entry


		# Pack the results together
		results_frame = tkinter.Frame(self.root,pady=20)
		results_frame.pack(side="top")
		results_frame.configure(background="white")
		submit_dafd_button = ttk.Button(results_frame, text='Run DAFD',command = self.runInterp)
		submit_dafd_button.pack(side="top")
		submit_fwd_button = ttk.Button(results_frame, text='Run Forward Model',command = self.runForward)
		submit_fwd_button.pack(side="top")
		self.results_label = tkinter.Label(results_frame)
		self.results_label.pack(side="top")
		self.results_label.configure(background="white")

		
		# Start GUI
		self.root.mainloop()


	def runInterp(self):
		"""Suggest design parameters based on given constraints and desired outputs"""

		# Get all of our constraints
		# It is quite possible to have no constraints if the user does not have a preference
		constraints = {}
		for param_name in self.di.input_headers:
			param_entry = self.entries_dict[param_name].get()
			if param_entry != "":
				# The constraint can either be a single value or a range
				if "-" in param_entry:
					# If it is a range x to y, the range is x-y
					pair = param_entry.split("-")
					wanted_constraint = (float(pair[0]),float(pair[1]))
				else:
					# If it is a single value x, the range is x-x
					wanted_constraint = (float(param_entry),float(param_entry))

				# Make sure that our constraints are in range. Just creates a warning if not.
				if wanted_constraint[0] <= self.di.ranges_dict[param_name][0]:
					tkinter.messagebox.showwarning("Out of range constraint",param_name + " was too low.")
				elif wanted_constraint[1] >= self.di.ranges_dict[param_name][1]:
					tkinter.messagebox.showwarning("Out of range constraint",param_name + " was too high.")
				constraints[param_name] = wanted_constraint


		# Regime must be 1 or 2 (dripping or jetting regime)
		regime_entry = self.entries_dict["regime"].get()
		if regime_entry != "":
			if regime_entry == "1" or regime_entry == "2":
				constraints["regime"] = float(regime_entry)
			else:
				tkinter.messagebox.showwarning("Regime must be either 1 or 2. Ignoring.")

		# Get the desired outputs
		# Note one can be left blank, in which case the interpolation model will simply operate on the other value's model
		desired_vals = {}
		for param_name in self.di.output_headers:
			param_entry = self.entries_dict[param_name].get()
			if param_entry != "":
				wanted_val = float(param_entry)
				if wanted_val >= self.di.ranges_dict[param_name][0] and wanted_val <= self.di.ranges_dict[param_name][1]:
					desired_vals[param_name] = wanted_val
				else:
					tkinter.messagebox.showwarning("Out of range desired value", param_name + " was out of range.")
					desired_vals[param_name] = wanted_val

		# Return and display the results
		results = self.di.runInterp(desired_vals,constraints)

		# Run Tolerance Test if Specified
		if bool(self.entries_dict["tolerance_test"].getvar(name="PY_VAR0")):
			from DAFD.tolerance_study.TolHelper import TolHelper
			tolerance = self.entries_dict["tolerance"].get()
			if tolerance == "":
				tolerance = 10
			tol_features = results.copy()
			del tol_features["point_source"]
			TH = TolHelper(tol_features, di=self.di, tolerance=float(tolerance))
			TH.run_all()
			TH.plot_all()
			TH.generate_report()

		print(self.di.runForward(results))
		self.results_label["text"] = "\n".join([x + " : " + str(results[x]) for x in self.di.input_headers])
	
	def runForward(self):
		"""Predict the outputs based on the chip geometry (Normal feed-forward network) """

		# Get all the chip geometry values
		features = {x: float(self.entries_dict[x].get()) for x in self.di.input_headers}
		results = self.di.runForward(features)
		self.results_label["text"] = "\n".join([x + " : " + str(results[x]) for x in results])

		# Run Tolerance Test if Specified
		if bool(self.entries_dict["tolerance_test"].getvar(name="PY_VAR0")):
			from DAFD.tolerance_study.TolHelper import TolHelper
			tolerance = self.entries_dict["tolerance"].get()
			if tolerance == "":
				tolerance = 10
			TH = TolHelper(features, di=self.di, tolerance=float(tolerance))
			TH.run_all()
			TH.plot_all()
			TH.generate_report()

#Executed when script is called from console
DAFD_GUI()
