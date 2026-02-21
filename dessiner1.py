import var1 as var
import fun1 as fun
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


continuer = np.array([True])
save = [-1]
colors0_1 = var.couleur/255
r0 = var.r - 3
t=0

traj = []
for _ in range(var.t_max):
    if not continuer[0]:
        break
    t+=var.dt
    #si on fait juste traj.append(var.sys), cela renvoie un pointeur tjr vers la même valeur
    traj.append(var.sys.copy())
    #fun.actual met a jour les coordonées de sys
    fun.actual(var.sys[None,None,:],var.sysv[None,None,:],var.sysa[None,None,:],t,var.dt,var.t_max,var.res,var.m,var.r,var.fixe,var.i_sat,save, continuer[None,None,:])

print(save)
traj_np = np.array(traj)

#on doit avoir une shape (2,N,len(t)) pour afficher
traj_np = np.transpose(traj_np,(2,1,0))

fig, axis = plt.subplots()

axis.set_xlim([0,var.long])
axis.set_ylim([0,var.haut])

axis.set_title("Simulation N-corps")
axis.set_aspect("equal")

points = []
for i in range(len(var.sys)):
    p, = axis.plot([], [], 'o',
                   markersize=r0[i],
                   color=colors0_1[i])
    points.append(p)


def update_data(frame):
    #dans anim.set_data, on doit avoir un tab de shape (2,N,len(t))
    for i, p in enumerate(points):
        p.set_data(
            [traj_np[0, i, frame]],
            [traj_np[1, i, frame]]
        )
    return points,


animation = FuncAnimation(
    fig=fig,
    func= update_data,
    frames= len(traj),
    interval = 25,
    )

#animation.save("nom.gif")
plt.show()