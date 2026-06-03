from integrateurs.base import BaseIntegrator
import numpy as np

class Leapfrog_integrator(BaseIntegrator):
    def __init__(self,scenario):
        self.scenario = scenario
        self.planetes = scenario.planetes
        self.nom = "Leapfrog"

    def calcul_force(self,k,i,res,G,d):
        """calcule la force qu'exerce p2 sur p1 divisé par la masse de p1 (donc l'acceleration due a la gravité)"""
        if d==0:
            return np.zeros_like(self.planetes[k].pos)
        F = []
        for c in range(len(self.planetes[0].pos)):
            F.append(self.planetes[i].masse *G *(self.planetes[i].pos[c] - self.planetes[k].pos[c])*(1/d**3) -res*self.planetes[k].vit[c])
        return np.array(F)
    
    def update_pos(self):
        for p in self.planetes:
            if p.fixe :
                continue
            p.pos += p.vit * self.scenario.dt + 0.5 * p.acc * self.scenario.dt**2
            p.acc_old = p.acc.copy()  # on sauvegarde a(t)

    def update_vit(self):
        for p in self.planetes:
            if p.fixe:
                continue
            p.vit += 0.5 * (p.acc_old + p.acc) * self.scenario.dt

    def step(self):
        # sys.shape = (len(sys),2) : on stocke un tableau couples de deux coordonées, on pose ci-dessous n = nombre de planètes
        n = len(self.planetes)
        save = -2
        continuer = True
        self.update_pos()

        # pour toutes les planètes du système, si elles ne sont pas fixes, on somme l'accélération liée aux autres planètes
        for k in range(n):
            sumk = np.zeros_like(self.planetes[0].pos)
            if self.planetes[k].fixe:
                continue
            for i in range(n):
                if k==i:
                    continue
                # d: distance entre la planète i et k
                d = self.planetes[i].dist(self.planetes[k])
                if d <= self.planetes[i].rayon + self.planetes[k].rayon:
                    if k == self.scenario.i_sat:
                        continuer=False
                        #si la position initiale a laquelle on lache le satelite est dans une des planètes, alors la couleur renvoyée est noire
                        if (self.scenario.t==0) and d <= self.planetes[i].rayon:
                            save = -1
                            return save,continuer
                        #si le satellite touche une planete, on renvoie la couleur associée à cette planète
                        else:
                            save = i
                            return save, continuer
                    else:
                        continue
                sumk += self.calcul_force(k, i,self.scenario.res,self.scenario.G,d)
            self.planetes[k].acc = sumk

        #actualisation de la position et de la vitesse
        self.update_vit()
        return save, continuer

    def get_positions(self):
        return np.array([planete.pos for planete in self.planetes])