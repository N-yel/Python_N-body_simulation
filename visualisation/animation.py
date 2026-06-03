import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core import couleur_to_couleur01

def animer(trajectoire,scenario):
    #on doit avoir une shape (2 ou 3,N,len(t)) pour afficher
    trajectoire = np.transpose(trajectoire,(2,1,0))
    dim = trajectoire.shape[0]

    if dim==2:
        fig, axis = plt.subplots()
        axis.set_xlim([-scenario.long/2,scenario.long/2])
        axis.set_ylim([-scenario.haut/2,scenario.haut/2])

        axis.set_xlabel("x (km)")
        axis.set_ylabel("y (km)")

        axis.set_aspect("equal")

        points = []
        traces = []

        couleurs = scenario.couleurs()
        rayons = scenario.rayons()
        for i in range(trajectoire.shape[1]):
            couleur = couleur_to_couleur01(couleurs[i])
            p, = axis.plot([], [], 'o',
                        markersize=rayons[i],
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
                    [trajectoire[0, i, frame]],
                    [trajectoire[1, i, frame]]
                )
                traces[i].set_data(
                    trajectoire[0, i, :frame],
                    trajectoire[1, i, :frame]
                )
            return points + traces


        animation = FuncAnimation(
            fig=fig,
            func= update_data,
            frames= trajectoire.shape[-1],
            interval = 25,
            )
        plt.show()
    if dim==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim(-scenario.long/2,scenario.long/2)
        ax.set_ylim(-scenario.haut/2,scenario.haut/2)
        ax.set_zlim(-scenario.profond/2,scenario.profond/2)

        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_zlabel("z (km)")


        ax.set_title("Simulation N-corps")
        ax.set_aspect("equal")

        points = []
        traces = []
        rayons = scenario.rayons()
        couleurs = scenario.couleurs()
        for i in range(trajectoire.shape[1]):
            couleur = couleur_to_couleur01(couleurs[i])
            p, = ax.plot([], [],[], 'o',
                        markersize=np.log10(rayons[i]),
                        color= couleur)
            points.append(p)
            trace, = ax.plot([], [],[], '-',
                                linewidth=1,
                                alpha=0.5,         # transparence de la trace
                                color=couleur)
            traces.append(trace)


        def update_data(frame):
            #dans anim.set_data, on doit avoir un tab de shape (2,N,len(t))
            for i in range(len(points)):
                points[i].set_data_3d(
                    [trajectoire[0,i, frame]],
                    [trajectoire[1,i, frame]],
                    [trajectoire[2,i, frame]]
                )
                traces[i].set_data_3d(
                    trajectoire[0, i, :frame],
                    trajectoire[1, i, :frame],
                    trajectoire[2, i, :frame]
                )
            return points + traces


        animation = FuncAnimation(
            fig=fig,
            func= update_data,
            frames= range(1,trajectoire.shape[2]),
            interval = 25,
            )

        plt.show()