import numpy as np
import matplotlib.pyplot as plt
import random as rand

# data size
obs = 500
# random parameters 
np.random.seed(10)
r1 = rand.uniform(-1.5,1.5)
r2 = rand.uniform(-1.5,1.5)
r3 = rand.uniform(-1.5,1.5)
r4 = rand.uniform(-1.5,1.5)

# inputs
class_zeros = np.random.multivariate_normal([0, 0], [[r1, r2], [r2, r1]], obs)
class_ones = np.random.multivariate_normal([0,5], [[r3, r4], [r4, r3]], obs)
bias = np.ones((2*obs, 1))
# organize
inputs = np.vstack((class_zeros, class_ones)).astype(np.float32)
inputs = np.hstack((bias, inputs)) #(obs,3)
# [ bi  class_zeros
#   as  class_ones]

# outputs
label_zeros = np.zeros((obs, 1))
label_ones = np.ones((obs, 1))
# organize
outputs = np.vstack((label_zeros, label_ones))
#[ label_zeros
#  label_ones ]

# merge
dataset = np.hstack((inputs,outputs))
# shuffle
np.random.shuffle(dataset)
# define
in_train = dataset[:,0:3]
out_train = dataset[:,3]

# perceptron
theta = np.random.randn(3) #(3,)
 
n = 0.15 # learning rate
t = 100 # interactions

trained = 0

# sigmoid function
def f0(x):
    return 1/(1+np.exp(-np.dot(x,theta.T))) # (obs,3)*(3,) = (obs,)

# plot
def plot_perceptron_train(theta):
    #plot line
    x1 = np.linspace(np.amin(inputs[:,1]),np.amax(inputs[:,1]),100)
    x2 = -(theta[0]+x1*theta[1])/theta[2] 
    
    #plot data
    plt.clf()
    plt.scatter((inputs[:, 1])[0:obs], (inputs[:, 2])[0:obs], c='b', marker='.', label='data 0')
    plt.scatter((inputs[:, 1])[obs:2*obs], (inputs[:, 2])[obs:2*obs], c='r', marker='X', label='data 1')
    plt.plot(x1, x2, c ='g', linestyle='solid',label='line_perceptron')
    plt.grid(linestyle='-.', linewidth=0.5)
    plt.legend()
    plt.draw()
    plt.pause(0.01)

# mouse click function to plot result from perceptron in graph
def onclick(event):
    if(trained):
        global ix, iy
        ix = event.xdata
        iy = event.ydata
        in_test = np.hstack((ix, iy))
        in_test = np.hstack((1,in_test))
        in_result = f0(in_test)
        print('Point clicked: ')
        print(round(ix,2),round(iy,2))
        print('Perceptron result: ')
        print(round(in_result,2))
        if(in_result >= 0.9):
            plt.plot(ix, iy, c ='m', marker='X')
        else:
            plt.plot(ix, iy, c='m', marker='.')
        plt.draw()

# assign event click to the figure
fig = plt.figure(1)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# update weights
for i in range(t):
    theta += -n*np.dot((f0(in_train)-out_train).T,in_train)
    # theta: (1,3); f^T: (1,obs); x: (obs,3) -> f^T*x: (1,3) = theta: (1,3)
    plot_perceptron_train(theta)
    if i==(t-1):
        print('Weights:')
        print(theta)
        print('Now you can click anywhere in the plot area and test the perceptron with untrained points')
        trained = 1
        plt.show(block=True)