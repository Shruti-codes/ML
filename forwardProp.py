import numpy as np

ip = np.array([2,3])

wt = {'n_0':np.array([1,1]),
	   	'n_1': np.array([-1,1]),
	   	'op': np.array([2,-1])}

n_0_val = (ip*wt['n_0']).sum()
n_0_op = np.tanh(n_0_val)
n_1_val = (ip*wt['n_1']).sum()
n_1_op = np.tanh(n_1_val)
hid = np.array([n_0_op,n_1_op])
print(hid)

op = (hid * wt['op']).sum()
print(op)