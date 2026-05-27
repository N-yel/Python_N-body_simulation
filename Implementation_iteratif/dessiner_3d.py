import numpy as np
from config import*
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


continuer = True
save = -1
t=0

traj = []
for _ in range(var.t_max):
    if not continuer:
        break
    t+=var.dt
    #si on fait juste traj.append(var.sys), cela renvoie un pointeur tjr vers la même valeur
    traj.append(np.array([planete.pos.copy() for planete in var.planetes]))
    #fun.actual met a jour les coordonées, vitesses, accélérations, de chacune des planètes dans le tab var.sys auquel l'on ajoute des dimensions
    save, continuer = fun.actual(var.planetes,var.t,var.res,var.i_sat,var.G)
    #print(var.planetes[1].pos)

# afficher la couleur de la planète vers laquelle le satellite est rentré en collision
#print(save)

traj_np = np.array(traj)
#on doit avoir une shape (3,N,len(t)) pour afficher
traj_np = np.transpose(traj_np,(2,1,0))


"""Tracé des données"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-var.long/2,var.long/2)
ax.set_ylim(-var.large/2,var.large/2)
ax.set_zlim(-var.profond/2,var.profond/2)

ax.set_title("Simulation N-corps")
ax.set_aspect("equal")

points = []
traces = []
for i in range(traj_np.shape[1]):
    p, = ax.plot([], [],[], 'o',
                markersize=np.log10(var.planetes[i].r),
                color= 'blue')
    points.append(p)
    trace, = ax.plot([], [],[], '-',
                        linewidth=1,
                        alpha=0.5,         # transparence de la trace
                        color='blue')
    traces.append(trace)





def update_data(frame):
    #dans anim.set_data, on doit avoir un tab de shape (2,N,len(t))
    for i in range(len(points)):
        points[i].set_data_3d(
            [traj_np[0,i, frame]],
            [traj_np[1,i, frame]],
            [traj_np[2,i, frame]]
        )
        traces[i].set_data_3d(
            traj_np[0, i, :frame],
            traj_np[1, i, :frame],
            traj_np[2, i, :frame]
        )
    return points + traces


animation = FuncAnimation(
    fig=fig,
    func= update_data,
    frames= range(1,traj_np.shape[2]),
    interval = 25,
    )

plt.show()
