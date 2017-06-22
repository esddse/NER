import matplotlib.pyplot as plt

epochs = [1,2,3,4,5,6,7,8,9,10, 11]
num = [73.44, 74.54, 79.79, 81.98, 79.55, 80.78, 83.03, 84.24, 82.74, 84.13, 85.92]

plt.plot(epochs, num)
plt.savefig('./training.jpg')