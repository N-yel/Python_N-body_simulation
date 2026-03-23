import numpy as np
import var_poo as var

class Planete :
    def __init__(self,pos,vit,acc,m,r,dt,fixe,couleur):
        self.pos = pos
        self.vit = vit
        self.acc = acc
        self.m = m
        self.r = r
        self.dt = dt
        self.fixe = fixe
        self.couleur = couleur
    
    def dist(self,p):
        return np.linalg.norm(self.pos - p.pos)



    def calcul_force(self,p,res):
        d = self.dist(p)
        if d <= self.r + p.r:
            return [0,0]

        Fx = self.m *var.G *(self.pos[0] - p.pos[0])*(1/d**3) -res*p.vit[0]
        Fy = self.m *var.G *(self.pos[1] - p.pos[1])*(1/d**3) -res*p.vit[1]
        return np.array([Fx,Fy])
        

    def update(self):
        self.vit += self.acc*self.dt
        self.pos += self.vit*self.dt

