from config import*
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


continuer = True
save = -1
t=0

def couleur_to_couleur01(rgb):
    return (rgb[0]/255,rgb[1]/255,rgb[2]/255)

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
#on doit avoir une shape (2,N,len(t)) pour afficher
traj_np = np.transpose(traj_np,(2,1,0))



fig, axis = plt.subplots()

axis.set_xlim([0,var.long])
axis.set_ylim([0,var.haut])

axis.set_title("Simulation N-corps")
axis.set_aspect("equal")

points = []
traces = []
for i in range(len(var.planetes)):
    couleur = couleur_to_couleur01(var.couleur[i])
    p, = axis.plot([], [], 'o',
                   markersize=var.planetes[i].r,
                   color= couleur)
    points.append(p)
    trace, = axis.plot([], [], '-',
                       linewidth=1,
                       alpha=0.5,         # transparence de la trace
                       color=couleur)
    traces.append(trace)





def update_data(frame):
    #dans anim.set_data, on doit avoir un tab de shape (2,N,len(t))
    for i in range(len(points)):
        points[i].set_data(
            [traj_np[0, i, frame]],
            [traj_np[1, i, frame]]
        )
        traces[i].set_data(
            traj_np[0, i, :frame],
            traj_np[1, i, :frame]
        )
    return points + traces


animation = FuncAnimation(
    fig=fig,
    func= update_data,
    frames= len(traj),
    interval = 25,
    )

#animation.save("simulation_terre_lune.gif", writer="pillow")
plt.show()