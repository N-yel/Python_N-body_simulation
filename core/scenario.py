import numpy as np
class Scenario:
    def __init__(self, planetes,long, haut,G,t_max,i_sat,profond=None, res=0,t=0, dt=1,nom = ""):
        self.planetes = planetes   # liste de Planete
        self.dt       = dt
        self.t = t
        self.t_max    = t_max
        self.G = G
        self.res      = res        # frottement
        self.i_sat    = i_sat      # indice satellite (-1 = aucun)
        self.long     = long
        self.haut     = haut
        self.profond = profond
        self.nom = nom

    # helpers
    def masses(self):
        return np.array([p.masse for p in self.planetes])

    def rayons(self):
        return np.array([p.rayon for p in self.planetes])

    def positions(self):
        return np.array([p.pos for p in self.planetes])

    def vitesses(self):
        return np.array([p.vit for p in self.planetes])

    def fixes(self):
        return np.array([p.fixe for p in self.planetes])

    def couleurs(self):
        return np.array([p.couleur for p in self.planetes] + [(255,255,255),(0,0,0)])

