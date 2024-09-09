
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import normal

sample=normal(loc=50,scale=5,size=1000)

plt.hist(sample)

#look like noremal
mean=sample.mean()
std=sample.std()

#now fit for normal dist
from scipy.stats import norm
dist= norm(mean,std)

# from certain range we are choosing point
val=np.linspace(sample.min(),sample.max(),100)

probabilities = [dist.pdf(value) for value in val]

plt.hist(sample,density=True)
plt.plot(val,probabilities)
