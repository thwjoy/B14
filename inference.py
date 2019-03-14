import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import time

def gauss_likelihood(x, mu, sigma):
    return np.sum(np.exp(-(x - mu)**2 / 2 * sigma**2) / math.sqrt(2 * 3.14 * sigma ** 2))

# number of data samples
num_samples = 10

# Let's assume our model has a sd of 1
# Activity, see if you can modify the code to perform inference on the sd
sigma_l = 1

# parameters of exisiting data distribution, remember it can be multimodal...
mu = 0
sigma = 1
mu_2 = 3
sigma_2 = 2

# our prior beliefs on what the parameters are
mu_p = 2
sigma_p = 1

## Creat arrays of data
x = np.linspace(-10, 10, 100) # This is really just used to plot the data
z = np.random.normal(mu, sigma, num_samples) # These are the data samples from the first mode
z2 = np.random.normal(mu_2, sigma_2, num_samples) # These are the data samples from the second mode
z = np.concatenate((z, z2)) # Concaternate to one distribution
p = stats.norm.pdf(x, mu_p, sigma_p) # Gaussian prior, computes the prior values for every x


# Plotting basics
plt.ion()
fig = plt.figure(figsize=(12,10))
fig.subplots_adjust(hspace=0.8)
ax = fig.add_subplot(311)
lk = fig.add_subplot(312)
ps = fig.add_subplot(313)
ax.set_ylim([0, 0.5])
lk.set_ylim([0, 5])
ps.set_ylim([0, 1])
ax.set_title('Data')
lk.set_title('Likelihood as a function of the mean of our approximate distribution')
ps.set_title('Posterior as a function of the mean of our approximate distribution')
ax.set_ylabel('p(.)')
lk.set_ylabel('p(x|u)')
ps.set_ylabel('p(u|x)')
ax.set_xlabel('x')
lk.set_xlabel('u')
ps.set_xlabel('u')


# plot initial data 
ax.plot(z, np.zeros_like(z), 'x', color='b')
lk.plot(z, np.zeros_like(z), 'x', color='b')
ps.plot(z, np.zeros_like(z), 'x', color='b')

# These are just place holders for posterior and likelihood values
likelihoods = np.zeros_like(x)
posteriors = np.zeros_like(x)
# Draw lines so we can update them later
prior, = ax.plot(x, p, color='r') 
model, = ax.plot(x, stats.norm.pdf(x, mu, sigma_l), color='b')
likelihood_line, = lk.plot(x, likelihoods, color='g')
posterior_line, = ps.plot(x, posteriors, color='m')
plt.legend([prior, model, likelihood_line, posterior_line],
            ["Prior", "Model", "Trace of Likelihood", "Trace of Posterior"],
            bbox_to_anchor=(1.1, 1.05))


# For each value of x, calculate the likelihood and posterior and then plot on a graph
for count, i in np.ndenumerate(x):
    # Get the likelihood and posterior
    likelihoods[count] = gauss_likelihood(z, i, sigma_l)
    posteriors[count] = gauss_likelihood(z, i, sigma_l) * p[count]
    # Update the graph
    model.set_ydata(stats.norm.pdf(x, i, sigma_l))
    likelihood_line.set_ydata(likelihoods)
    posterior_line.set_ydata(posteriors)
    fig.canvas.draw()
    time.sleep(0.05)

plt.savefig('./plot.png')

