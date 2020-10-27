import csv
import torch
from utils.options import args_parser
from SV_utils import powerset #tool box fo FL_SV
import sys
import matplotlib.pyplot as plt 
import numpy as np
x_axs = list(range(0,10))
x_axs = [x/10 for x in x_axs]
# print(x_axs)
exact   = [71.59, 72.98, 72.00, 69.86, 66.03, 63.32, 51.82, 47.77, 44.17, 32.10]
beihang = [71.59, 67.49, 64.72, 63.28, 63.19, 60.23, 51.56, 44.17, 44.09, 32.10]

plt.plot(x_axs,exact,label = 'exact')
plt.plot(x_axs,beihang,label = 'beihang')
plt.legend(['Exact','Beihang'])
plt.xlabel("Frac of least important clients removed")
plt.ylabel("Testing acc on trained FL model")
plt.title('Testing Acc of deleting least valuable clients')
# plt.show()
plt.savefig('save/deleting_result.png')