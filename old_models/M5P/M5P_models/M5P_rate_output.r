#Usage: Rscript M5P_rate_output.r > generation_rate.tree 
dat = read.csv("../MicroFluidics_Random.csv")
library("RWeka")
model = M5P(generation_rate~orifice_size+aspect_ratio+width_ratio+normalized_orifice_length+normalized_oil_input_width+normalized_water_input_width+capillary_number+flow_rate_ratio,data = dat,control = Weka_control(N=TRUE))

print(model)
