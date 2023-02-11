import numpy as np 
from matplotlib import pyplot as plt 

x = np.linspace(0,10, num = 100) 
y = x ** 2 + (np.sin(100*x))**3 + 5

print(y)

plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.show()

