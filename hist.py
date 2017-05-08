import numpy as np
import matplotlib as plt
s=np.random.uniform(6,7,25)

plt.figure(10)

plt.hist(s,4)

smin=s.min()
smax=s.max()

bin_width=(smax-smin)/4

first_left=smin
second_left=smin+bin_width
third_left=second_left+bin_width
fourth_left=smin+bin_width*3
bin_left=np.array([first_left,second_left,third_left,fourth_left])

first_height=np.size(np.where((s>=smin) & (s<second_left)))
second_height=np.size(np.where((s>=second_left) & (s<third_left)))
third_height=np.size(np.where((s>=third_left) & (s<fourth_left)))
fourth_height=np.size(np.where((s>=fourth_left) & (s<=smax)))

bin_height=np.array([first_height,second_height,third_height,fourth_height])

plt.figure(11)
plt.subplot(2,1,1)

plt.bar(bin_left,bin_height,bin_width)

plt.subplot(2,1,2)

plt.bar(bin_left,bin_height,bin_width-0.1)

plt.figure(12)

plt.bar(bin_left,bin_height/np.size(s),bin_width-0.1)

np.cumsum(bin_height/np.size(s))
pdf=bin_height/np.size(s)
cdf=np.cumsum(pdf)

plt.figure(12)
plt.subplot(3,1,2)
plt.bar(bin_left,cdf,bin_width-0.1)

plt.subplot(3,1,3)
plt.bar(bin_left,bin_height,bin_width)