"""
This file contains all the functions that we will use for simulation.
Comments are in French throughout the rest of the document.
"""
import numpy as np
import var1 as var
from time import perf_counter

def actual(tab_sys,tab_sysv,tab_sysa,t,dt,res,m,r,fixe,i_sat,save, tab_continuer):
    """
    Va actualiser tab_sys, tab_sysv, tab_sysa en suivant les lois de Newton
    Arguments:
        tab_sys : np.array, dtype=float , shape=(long,haut,N,2) va stocker la position avec les coordonées (x,y) de chaque planète d'un système à N-corps, et ça pour chaque point d'une matrice de taille (long x haut)
        tab_sysv : np.array, dtype=float , shape=(long,haut,N,2) va stocker la vitesse avec les coordonées (vx,vy) de chaque planète d'un système à N-corps, et ça pour chaque point d'une matrice de taille (long x haut)
        tab_sysa : np.array, dtype=float , shape=(long,haut,N,2) va stocker l'accélération avec les coordonées (ax,ay) de chaque planète d'un système à N-corps, et ça pour chaque point d'une matrice de taille (long x haut)
        t: temps
        dt: intervalle de temps entre 2 calcul, plus il est petit, plus la simulation est réaliste, mais plus le cout sera grand
        res: Correspond à un forettement fluide, plus il sera élevé plus les planètes convergeront vite
        m: np.array, dtype=float, shape=(N) va stocker la masse de chacune des planètes, elle ne change pas d'un point a l'autre de la matrice (long,large)
        r: np.array, dtype=float, shape=(N) va stocker le rayon de chacune des planètes, il ne change pas d'un point a l'autre de la matrice (long,large)
        fixe: np.array, dtype=bool, shape =(N) va stocker si l'on actualise la position de la i-ème planète ou pas. Si fixe[i]==True, alors on ne modifiera ni sa vitesse, ni sa position.
        i_sat: int correspond à l'indice du satellite, si on veut simuler un système sans satellite, on le met à -1. On arreter d'actualiser si il y a une collision entre i_sat et une autre planète
        save: np.array, dtype=int, shape = (long,haut) initialisé avec que des -1, il renverra l'indice de la planète qui rentre en collision avec le sattelite i_sat
        tab_continuer: np.array, dtype=bool, shape = (long,haut) initialisé avec que des TRUE, Il passera à False en (i,j) dès qu'il y a une collision dans cette case
    
    Returns:
        Void : on actualise tab_sys, tab_sysv et tab_sysa, mais on ne return rien
    """
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
        mask_continuer = True
    else:
        d_isat = d[...,i_sat,:]
        r_isat = rayons[...,i_sat,:]
        touche_isat = d_isat <= r_isat

        # touche_isat.shape = (long,haut,n), np.all(, axis = -1) va renvoyer pour chaque point si il y a un des i_sat qui est en collision avec une planète
        mask_continuer = np.all(~touche_isat, axis = -1)


        if(t <=1):
            print(np.count_nonzero(~mask_continuer))
            #np.where(condition, if, else) va renoyer un tableau de la même shape que le tableau de booléen condition, qui contient la variable if dans les cases ou la condition est respectée, else sinon
            #donc tout les endroits ou on ne continue pas, on met argmax (vu qu'il y a a priori une seule collision a la fois, argmax va renvoyer le seul 1(True)), sinon on laisse save
            save[:] = np.where(~mask_continuer, -1,save)
        else:
            save[:] = np.where(~mask_continuer, np.argmax(touche_isat, axis=-1),save)

    tab_continuer[:] = tab_continuer & mask_continuer
    acc_tab =-var.G*(tab_continuer[...,None,None]*mask*m[None,:]*(1/d**3))[...,None]*diff
    #print(acc_tab)
    tab_sysa[:] = np.sum(acc_tab, axis=-2) -res*tab_sysv

    # on actualise la vitesse en fonction de l'accélération
    tab_sysv[:]+= dt *tab_sysa
    # on actualise la position en fonction de la vitesse
    tab_sys[:] += dt*tab_sysv
    return


def placer_satellite(tab_sys, i_sat):
    """
    va modifier une grille déjà existente tab_sys, et pour mettre pour chaque point (i,j) de (long, large), va mettre tab_sys[long][large][i_sat] à (i,j): on déplace chacun des satellites à un point différent de la grille
    Arguments:
        tab_sys : np.array, dtype=float , shape=(long,haut,N,2)
        i_sat : int correspond à l'indice du satellite, si on veut simuler un système sans satellite, on le met à -1.
    Returns:
        void : on ne fait que modifier tab_sys, on ne renvoie rien
    """
    if i_sat==-1:
        raise ValueError("On ne peut pas creer une grille si on n'a pas de satellites")
    
    long, large = tab_sys.shape[0], tab_sys.shape[1]

    I, J = np.meshgrid(np.arange(long), np.arange(large), indexing="ij")  # shape (long,large)
    tab_sys[..., i_sat, 0] = I
    tab_sys[..., i_sat, 1] = J


def creer_grille_sys_dense(sys, long, haut,i_sat):
    """
    va mettre dans chaque case d'une matrice (long, haut) la tab nommé sys, puis va modifier la position de chaque i_sat en utilisant la fonction placer_satellite
    Arguments:
        sys : np.array, dtype=float , shape=(N,2)
        long : int représente la longueur du tableau renvoyé
        haut : int représente la largeur du tableau renvoyé
        i_sat : int correspond à l'indice du satellite, si on veut simuler un système sans satellite, on le met à -1.
    Returns:
        tab_sys : np.array, dtype=float , shape=(long,haut,N,2)
    """
    # np.broadcast_to va "rajouter des dimensions"
    tab_sys = np.broadcast_to(sys[None, None, :, :], (long, haut) + sys.shape).copy()
    placer_satellite(tab_sys,i_sat)
    #print(tab_sys[1,42,i_sat])
    return tab_sys

def creer_grille_indep(s,long,haut):
    """
    va mettre dans chaque case d'une matrice (long, haut) la tab nommé s, la différence avec la fonction précédente c'est qu'on ne modifie aucune information. Uttile pour copier sysv et sysa
    Arguments:
        s : np.array, dtype=float , shape=(N,2)
        long : int représente la longueur du tableau renvoyé
        haut : int représente la largeur du tableau renvoyé
    Returns:
        tab : np.array, dtype=float , shape=(long,haut,N,2)
    """
    tab = np.broadcast_to(s[None, None, :, :], (long, haut) + s.shape).copy()
    return tab

#va renvoyer un tableau de dimension (long,haut) avec la couleur de la planète vers laquelle le satellite d'indice i_sat va atterir en fonction de sa position
def calculer(sys,sysv,long,haut,dt,t_max,res,i_sat,m,r,fixe):
    """
    Fonction très lourde, peut mettre plusieures heures à s'executer selon la taille du tableau, tmax, et dt.
    La fonction calculer simule l’évolution temporelle d’un satellite placé initialement en chaque point d’une matrice (long x haut), et détermine pour chaque position s’il entre en collision avec une planète ou reste en orbite.
    Elle renvoie une matrice indiquant le résultat final de chaque trajectoire.
    Arguments:
        sys: np.array, dtype=float, shape=(N,2) système initial
        sysv: np.array, dtype=float, shape=(N,2)
        long: int
        haut: int
        dt: float
        t_max: float
        res: float
        i_sat: int correspond à l'indice du satellite, si on veut simuler un système sans satellite, on le met à -1.
        m: np.array, dtype=float, shape=(N) va stocker la masse de chacune des planètes, elle ne change pas d'un point a l'autre de la matrice (long,large)
        r: np.array, dtype=float, shape=(N) va stocker le rayon de chacune des planètes, il ne change pas d'un point a l'autre de la matrice (long,large)
        fixe: np.array, dtype=bool, shape =(N) va stocker si l'on actualise la position de la i-ème planète ou pas. Si fixe[i]==True, alors on ne modifiera ni sa vitesse, ni sa position.
    Returns:
        var.couleur[save]: np.array, dtype=int, shape=(long,haut,3) renvoie un tableau qui, pour chaque case (i,j), renvoie la couleur sous format RGB de la planète vers lequel le satellite est rentré en collision.
        Si ne rentre pas en collision avant tmax, alors on renvoie (42,42,42)
    """
    t=0
    start0 = perf_counter()

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
        actual(tab_sys,tab_sysv,tab_sysa,t,dt,res,m,r,fixe,i_sat,save, tab_continuer)
        t+=dt

    save[tab_continuer] = -2
    temps = perf_counter() - start0
    print(f"l'algorithme aura mis {round(temps/60)} min et {round(temps%60)} secondes")
    return var.couleur[save]