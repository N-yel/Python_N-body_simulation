from integrateurs.base import BaseIntegrator
from core.utilitaires import creer_grille_sys_dense,creer_grille_indep
import numpy as np

#Ne marche qu'en 2 dimensions
class Parallel_Integrator(BaseIntegrator):

    def __init__(self,scenario):
        self.scenario = scenario
        self.tab_continuer = np.full((scenario.long,scenario.haut),True,dtype = bool)
        self.tab_pos = creer_grille_sys_dense(scenario.positions(),scenario.long,scenario.haut,scenario.i_sat)
        self.tab_vit = creer_grille_indep(scenario.vitesses(),scenario.long,scenario.haut)
        self.tab_acc = np.zeros(self.tab_pos.shape, dtype = float)
        self.tab_acc_old = np.zeros(self.tab_pos.shape, dtype = float)
        self.masses = scenario.masses()
        self.rayons = scenario.rayons()
        self.fixes = scenario.fixes()
        self.nom = "parallel_integrator"
    
    def update_pos(self):
        self.tab_pos[:] += self.tab_vit * self.scenario.dt + 0.5 * self.tab_acc * self.scenario.dt**2
        self.tab_acc_old[:] = self.tab_acc.copy()  # on sauvegarde a(t)

    def update_vit(self):
        self.tab_vit[:] += 0.5 * (self.tab_acc_old + self.tab_acc) * self.scenario.dt

    def get_positions(self):
        return self.tab_pos[0]

    def step(self):
        tab_save = np.full((self.scenario.long,self.scenario.haut),-2)
        self.update_pos()

        diff = self.tab_pos[...,:,None,:] - self.tab_pos[...,None,:,:] #diff.shape = (long,haut,n,n,2) pour n planètes
        d = np.linalg.norm(diff, axis = -1) #d.shape = (long,large,n,n), on a fait la norme entre de chaque coordonée du dernière axe
        n = d.shape[-1]
        d[...,np.arange(n),np.arange(n)] = np.inf
        rayons = self.rayons[...,:,None] + self.rayons[...,None,:]
        mask1 = (d[...,:,:] > rayons) # si il y a une collision entre i et j, mask1[i][j] = False   mask1.shape = (n,n) sans pour autant arreter la simulation, cela sert juste à éviter les divergences vers l'infini de la force lorsque le rayon tend vers 0
        mask2 = ~self.fixes[:,None]# si le i est fixe, alors \forall j mask2[i][j] = False
        # # lors du calcul de la force, on va multiplier par mask de sorte a ne pas calculer la force lorsqu'il y a un problème
        mask = mask1 & mask2

        if self.scenario.i_sat == -1:
            mask_continuer = True
        else:
            d_isat = d[...,self.scenario.i_sat,:]
            r_isat = rayons[...,self.scenario.i_sat,:]
            touche_isat = d_isat <= r_isat

            # touche_isat.shape = (long,haut,n), np.all(, axis = -1) va renvoyer pour chaque point si il y a un des i_sat qui est en collision avec une planète
            mask_continuer = np.all(~touche_isat, axis = -1)

            if(self.scenario.t ==0):
                #np.where(condition, if, else) va renoyer un tableau de la même shape que le tableau de booléen condition, qui contient la variable if dans les cases ou la condition est respectée, else sinon
                #donc tout les endroits ou on ne continue pas, on met argmax (vu qu'il y a a priori une seule collision a la fois, argmax va renvoyer le seul 1(True)), sinon on laisse save
                tab_save[:] = np.where(~mask_continuer, -1,tab_save)
            else:
                tab_save[:] = np.where(~mask_continuer, np.argmax(touche_isat, axis=-1),tab_save)

            self.tab_continuer[:] = self.tab_continuer & mask_continuer

            d_safe = np.where(mask, d, np.inf)  # là où mask=False, on met inf → 1/inf = 0
            F =-self.scenario.G*(self.tab_continuer[...,None,None]*mask*self.masses[None,:]*(1/d_safe**3))[...,None]*diff
            self.tab_acc[:] = np.sum(F, axis=-2) -self.scenario.res*self.tab_vit

            self.update_vit()
            return tab_save


