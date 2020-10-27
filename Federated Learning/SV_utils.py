#tool box for Federated Learning shapley value

from itertools import permutations 



def powerset(s):
    x = len(s)
    to_return = []
    for i in range(1 << x):
    	to_return.append([s[j] for j in range(x) if (i & (1 << j))])
    return to_return


