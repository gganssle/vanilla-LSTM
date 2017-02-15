'''
This script randomizes the order of the feature vectors
in a data file, so that a trainer doesn't have to 
proceed in a sequential order.
'''

import numpy as np
from random import shuffle

# init
vlen = 5
vnm = 999
ln = 'labels'
dn = 'data' 

old = np.zeros((vnm, vlen))
new = np.zeros((vnm, vlen))

# random indices
idx = np.zeros(vnm)
for i in range(vnm):
	idx[i] = i

shuffle(idx)

# labels
	# read in data
with open(ln, 'r') as lf:
	for i in range(vnm):
		for j in range(vlen):
			temp = lf.readline()
			old[i,j] = temp

	# randomize vectors
for i in range(vnm):
	new[i] = old[int(idx[i])]

	# write out
with open(ln, 'w') as lf:
	for i in range(vnm):
		for j in range(vlen):
			lf.write(str(int(new[i][j])))
			lf.write("\n")

# data 
	# read in data
with open(dn, 'r') as df:
	for i in range(vnm):
		for j in range(vlen):
			temp = df.readline()
			old[i,j] = temp

	# randomize vectors
for i in range(vnm):
	new[i] = old[int(idx[i])]

	# write out
with open(dn, 'w') as df:
	for i in range(vnm):
		for j in range(vlen):
			df.write(str(int(new[i][j])))
			df.write("\n")
