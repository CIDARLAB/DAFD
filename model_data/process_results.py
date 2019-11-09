""" Unimportant utility script that was quickly made to reduce manual work in Excel"""

import sys

value_dict = {}

with open(sys.argv[1],"r") as f:
	for line in f:
		if ":" in line:
			entry = line.split(":")[0]
			value = line.split(":")[1].strip()
			if entry not in value_dict:
				value_dict[entry] = []
			value_dict[entry].append(value)

for entry in sorted(value_dict.keys()):
	print(entry + "," + ",".join(value_dict[entry]))# + "," + str(sum(map(float,value_dict[entry]))/len(value_dict[entry])))
				
