

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal

from scipy.stats import norm

sample1=normal(loc=20,scale=5,size=300)
sample2=normal(loc=40,scale=5,size=600)
sample=np.hstack((sample1,sample2))

plt.hist(sample)

#KDE
from sklearn.neighbors import KernelDensity
model=KernelDensity(kernel='gaussian',bandwidth=5)
sample=sample.reshape(len(sample),1)
model.fit(sample)

val=np.linspace(sample.min(),sample.max(),100)
#convert to 2D
val=val.reshape(len(val),1)

prob=model.score_samples(val)
# log value is given ie exp use
prob=np.exp(prob)

plt.hist(sample,density=True)
plt.plot(val,prob)

import seaborn as sns
sns.kdeplot(sample.reshape(900),)
