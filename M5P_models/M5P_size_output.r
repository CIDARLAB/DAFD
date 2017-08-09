#Usage: Rscript M5P_size_output.r > droplet_size.tree 
dat = read.csv("../MicroFluidics_Random.csv")
library("RWeka")
model = M5P(droplet_size~orifice_size+aspect_ratio+width_ratio+normalized_orifice_length+normalized_oil_input_width+normalized_water_input_width+capillary_number+flow_rate_ratio,data = dat,control = Weka_control(N=TRUE))

print(model)
