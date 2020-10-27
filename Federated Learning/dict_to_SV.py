import csv
import torch
from utils.options import args_parser
from SV_utils import powerset #tool box fo FL_SV
import sys
import matplotlib.pyplot as plt 
import numpy as np
args = args_parser()

def remove_list_indexed(removed_ele, original_l, ll):
    new_original_l = []
    for i in original_l:
        new_original_l.append(i)
    for i in new_original_l:
        if i == removed_ele:
            new_original_l.remove(i)
    for i in range(len(ll)):
        if set(ll[i]) == set(new_original_l):
            return i
    return -1

def find_without_list(user,subset,all_subset):
	for s in all_subset:
		u = []
		u.append(user)
		if list(set(subset) - set(u)) == s:
			if s != None:
				return s	
def normalize(lst):
    s = sum(lst)
    return map(lambda x: float(x)/s, lst)

def get_sv(file_name):
	with open(file_name) as csv_file:
	    reader = csv.reader(csv_file)
	    mydict = dict(reader)

	for key in mydict:
		mydict[key] = mydict[key][6:]
		mydict[key] = float(mydict[key][1:-1])
		key = tuple(key)

	# print(mydict)
	# print(mydict[str(tuple([0,1]))])
	client_lst = list(range(0, args.num_users))
	all_subset = powerset(client_lst)  #record all subset of a client list
	all_subset.remove([])
	# print(tuple(all_subset[4]))

	# sys.exit(0)
	record_lst = []
	for user in range(args.num_users):
		sv = 0.0
		for subset in all_subset:
			# print(user,subset)
			if user in subset:
				without_set = find_without_list(user,subset,all_subset)
				if without_set:
					with_util = mydict[str(tuple(subset))]
					without_util = mydict[str(tuple(without_set))]
					# sv += abs(with_util - without_util)
					sv += with_util - without_util
		record_lst.append(sv)

	# print(record_lst)
	return record_lst


if __name__ == '__main__':
	f1 = 'exact_dict.csv'
	f2 = 'beihang_dict.csv'
	exact = get_sv(f1)
	beihang = get_sv(f2)
	exact = [round(i/sum(exact),5) for i in exact]
	beihang = [round(i/sum(beihang),5) for i in beihang]
	print(exact)
	print(beihang)

	seq1 = sorted(exact)
	index1 = [exact.index(v) for v in seq1]
	seq2 = sorted(beihang)
	index2 = [beihang.index(v) for v in seq2]	

	print(index1)
	print(index2)
	x_axs = range(0,10)
	plt.plot(x_axs,exact, label = 'exact')
	plt.plot(x_axs,beihang, label = 'beihang')
	plt.title('SV of exact and beihang')
	plt.show()