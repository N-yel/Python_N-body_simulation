import numpy as np

class Planete :
    def __init__(self,pos,vit,acc,m,r,dt,fixe):
        self.pos = pos
        self.vit = vit
        self.acc = acc
        self.acc_old = np.array([0 for _ in pos],dtype=float)
        self.m = m
        self.r = r
        self.dt = dt
        self.fixe = fixe
    
    # renvoie en texte la représentation d'une planète (utile pour le fichier csv)
    def __repr__(self):
        return f"pos={self.pos}, vit={self.vit}, m={self.m}, r={self.r}, dt={self.dt}, fixe={self.fixe}"

    def dist(self,p):
        return np.linalg.norm(self.pos - p.pos)
    
    def update_pos(self):
        self.pos += self.vit * self.dt + 0.5 * self.acc * self.dt**2
        self.acc_old = self.acc.copy()  # on sauvegarde a(t)

    def update_vit(self):
        # Étape 3 : v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        self.vit += 0.5 * (self.acc_old + self.acc) * self.dt
    



class Quadtree:
    def __init__(self, centre, taille):
        self.centre = np.array(centre, dtype=float)
        self.taille = taille          # demi-largeur du carré
        self.mtot = 0.0
        self.centre_masse = np.array([0.0, 0.0])
        self.nb_planetes = 0
        self.fils = [None, None, None, None]  # [hg, hd, bg, bd]
        self.planete = None                   # seulement si nœud feuille



    def get_quadrant(self, pos):
        """
        Retourne l'index du quadrant (0=hg, 1=hd, 2=bg, 3=bd)
        et le centre du fils correspondant.
        """
        cx, cy = self.centre
        quart = self.taille / 2

        if pos[0] < cx:      # gauche
            if pos[1] >= cy: # haut
                idx = 0
                new_centre = [cx - quart, cy + quart]
            else:            # bas
                idx = 2
                new_centre = [cx - quart, cy - quart]
        else:                # droite
            if pos[1] >= cy: # haut
                idx = 1
                new_centre = [cx + quart, cy + quart]
            else:            # bas
                idx = 3
                new_centre = [cx + quart, cy - quart]

        return idx, new_centre

    def est_feuille(self):
        return all(f is None for f in self.fils)

    def ajouter_planete(self, planete):
        #Mise à jour masse et barycentre
        #on calcule dans un premier temps la moyenne des position pondérée par la masse, puis on met à jour mtot pour enfin divisé la moyenne par mtot
        self.centre_masse = (self.centre_masse * self.mtot + planete.pos * planete.m)
        self.mtot += planete.m
        self.centre_masse /= self.mtot
        self.nb_planetes += 1

        #Deux cas de base: si il y a une feuille ou si il n'y en a aucune
        #Cas 1 : nœud vide -> on place la planète ici 
        if self.nb_planetes == 1:
            self.planete = planete
            return

        #Cas 2 : nœud feuille avec déjà une planète -> on subdivise
        if self.est_feuille():
            # On replace l'ancienne planète dans le bon fils
            ancienne = self.planete
            self.planete = None

            idx, new_centre = self.get_quadrant(ancienne.pos)
            self.fils[idx] = Quadtree(new_centre, self.taille / 2)
            self.fils[idx].ajouter_planete(ancienne)

        #Cas 3 : nœud interne (ou après subdivision) -> on descend
        idx, new_centre = self.get_quadrant(planete.pos)
        if self.fils[idx] is None:
            self.fils[idx] = Quadtree(new_centre, self.taille / 2)
        self.fils[idx].ajouter_planete(planete)
    
    def ajouter_planetes(self,planetes):
        for planete in planetes:
            self.ajouter_planete(planete)


    def calcul_force(self, planete, theta, G):
        # Nœud vide
        if self.nb_planetes == 0:
            return np.array([0.0, 0.0])

        # On ne calcule pas la force d'une planète sur elle-même
        if self.nb_planetes == 1 and self.planete is planete:
            return np.array([0.0, 0.0])

        d = np.linalg.norm(planete.pos - self.centre_masse)
        s = self.taille * 2

        # Nœud externe OU critère d'approximation satisfait
        if self.nb_planetes == 1 or (s / d) < theta:
            delta = self.centre_masse - planete.pos
            force = G * planete.m * self.mtot / d**3 * delta
            return force

        # Sinon on descend dans les fils
        else:
            force = np.array([0.0, 0.0])
            for fils in self.fils:
                if fils is not None:
                    force += fils.calcul_force(planete, theta, G)
            return force



    
