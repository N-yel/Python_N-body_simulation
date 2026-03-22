"""just so I can get used to matplotlib"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t= np.linspace(0,10,100)
x1 = np.cos(t)
y1 = np.sin(t)
x2 = np.sin(t)
y2 = np.cos(t)
x3 = 0.1*t
y3 = 0.1*t
tab = np.array([[x1,x2,x3],[y1,y2,y3]])
fig, axis = plt.subplots()

print(f"la shape sera{tab.shape}")
axis.set_xlim([-2,2])
axis.set_ylim([-2,2])

animated_plot1, = axis.plot([],[],'o',markersize=5,color='red')


def update_data(frame):
    animated_plot1.set_data(tab[...,frame])

    return animated_plot1,


animation = FuncAnimation(
    fig=fig,
    func= update_data,
    frames= len(t),
    interval = 25,
    )

#animation.save("nom.gif")
plt.show()