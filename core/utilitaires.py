import numpy as np
def couleur_to_couleur01(rgb):
    return (rgb[0]/255,rgb[1]/255,rgb[2]/255)



def _placer_satellite(tab_sys, i_sat):
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
    
    long, haut = tab_sys.shape[0], tab_sys.shape[1]

    I, J = np.meshgrid(np.arange(long) - long//2, np.arange(haut) - haut //2, indexing="ij")  # shape (long,large)



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
    _placer_satellite(tab_sys,i_sat)
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





