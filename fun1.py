import numpy as np
import var1 as var
from time import perf_counter

"""
Exemple utilisation np.meshgrid
x = np.array([0, 1, 2])
y = np.array([10, 20])
X, Y = np.meshgrid(x, y)

On va avoir 
X =
[[0 1 2]
 [0 1 2]]

Y =
[[10 10 10]
 [20 20 20]]

de plus, np.arrange(long) = [0,1,...,long -1]
"""

def placer_satellite(tab_sys, i_sat):
    long, large = tab_sys.shape[0], tab_sys.shape[1]

    I, J = np.meshgrid(np.arange(long), np.arange(large), indexing="ij")  # shapes (long,large)
    tab_sys[..., i_sat, 0] = I
    tab_sys[..., i_sat, 1] = J


def creer_grille_sys_dense(sys, long, haut,i_sat):
    # np.broadcast_to va "rajouter des dimensions"
    tab_sys = np.broadcast_to(sys[None, None, :, :], (long, haut) + sys.shape).copy()
    placer_satellite(tab_sys,i_sat)
    #print(tab_sys[1,42,i_sat])
    return tab_sys




def creer_grille_indep(s,long,haut):
    tab = np.broadcast_to(s[None, None, :, :], (long, haut) + s.shape).copy()
    return tab


def actual(tab_sys,tab_sysv,tab_sysa,t,dt,t_max,res,m,r,fixe,i_sat,save, tab_continuer):

    # sys[:,None,:] va 'rajouter une dimension' à l'endroit du None ce qui va permettre de sommer les tableaux
    diff = tab_sys[...,:,None,:] - tab_sys[...,None,:,:] #diff.shape = (long,haut,n,n,2) pour n planètes

    d = np.linalg.norm(diff, axis = -1) #d.shape = (long,large,n,n), on a fait la norme entre de chaque coordonée du dernière axe

    n = d.shape[-1]
    #pour eviter une division par 0, lorsqu'on est sur la diagonale de la matrice, on dit que la distance est de infty. Dans cette version du code, sachant que d.shape = (long,haut,n,n), on ne peut plus faire np.fill_diagonal, on se débrouille alors comme on peut
    d[...,np.arange(n),np.arange(n)] = np.inf


    rayons = r[...,:,None] + r[...,None,:]
    mask1 = (d[...,:,:] > rayons) # si il y a une collision entre i et j, mask1[i][j] = False   mask1.shape = (n,n)
    mask2 = ~fixe[:,None]# si le i est fixe, alors \forall j mask2[i][j] = False
    # # lors du calcul de la force, on va multiplier par mask de sorte a ne pas calculer la force lorsqu'il y a un problème
    mask = mask1 & mask2

    if i_sat == -1:
        mask_continuer1 = True
    else:
        d_isat = d[...,i_sat,:]
        r_isat = rayons[...,i_sat,:]
        touche_isat = d_isat <= r_isat

        # touche_isat.shape = (long,haut,n), np.all(, axis = -1) va renvoyer pour chaque point si il y a un des i_sat qui est en collision avec une planète
        mask_continuer1 = np.all(~touche_isat, axis = -1)


        if(t <=1):
            print(np.count_nonzero(~mask_continuer1))
            #np.where(condition, if, else) va renoyer un tableau de la même shape que le tableau de booléen condition, qui contient la variable if dans les cases ou la condition est respectée, else sinon
            #donc tout les endroits ou on ne continue pas, on met argmax (vu qu'il y a a priori une seule collision a la fois, argmax va renvoyer le seul 1(True)), sinon on laisse save
            save[:] = np.where(~mask_continuer1, -1,save)
        else:
            save[:] = np.where(~mask_continuer1, np.argmax(touche_isat, axis=-1),save)


    mask_continuer = mask_continuer1
    tab_continuer[:] = tab_continuer & mask_continuer
    acc_tab =-var.G*(tab_continuer[...,None,None]*mask*m[None,:]*(1/d**3))[...,None]*diff
    #print(acc_tab)
    tab_sysa[:] = np.sum(acc_tab, axis=-2) -res*tab_sysv

    # on actualise la vitesse en fonction de l'accélération
    tab_sysv[:]+= dt *tab_sysa
    # on actualise la position en fonction de la vitesse
    tab_sys[:] += dt*tab_sysv
    return


#va renvoyer un tableau de dimension (long,haut) avec la couleur de la planète vers laquelle le satellite d'indice i_sat va atterir en fonction de sa position
def calculer(sys,sysv,long,haut,dt,t_max,res,i_sat,m,r,fixe):
    t=0
    start0 = perf_counter()
    #img: va stocker en la couleur en fct des conditionss initiales
    save = np.full((long,haut),-1,dtype = int)
    tab_continuer = np.full((long,haut),True,dtype = bool)
    tab_sys = creer_grille_sys_dense(sys,long,haut,i_sat)
    tab_sysv = creer_grille_indep(sysv,long,haut)
    tab_sysa = np.zeros(tab_sys.shape, dtype = float)
    while t <= t_max:
        if t%10==0:
            nb_fini = np.count_nonzero(~tab_continuer)
            #on arrete si on a trouvé 95% des trajectoires ont été calculés : il ne reste plus que les trajectoires périodiques
            if nb_fini == long*haut*0.95:
                break
            print(t, nb_fini)
        actual(tab_sys,tab_sysv,tab_sysa,t,dt,t_max,res,m,r,fixe,i_sat,save, tab_continuer)
        t+=dt

    save[tab_continuer] = -2
    temps = perf_counter() - start0
    print(f"l'algorithme aura mis {round(temps/60)} min et {round(temps%60)} secondes")
    return var.couleur[save]