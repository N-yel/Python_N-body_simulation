import numpy as np
from core import Planete, Scenario

p0 = Planete(pos=np.array([-100,0],dtype=float),
                    vit=np.array([0,0],dtype=float),
                    masse=100,
                    rayon=10,
                    couleur=(255,0,0),
                    fixe=True,
                    nom="p0"
                    )

p1 = Planete(pos=np.array([100,0],dtype=float),
                    vit=np.array([0,0],dtype=float),
                    masse=100,
                    rayon=10,
                    couleur=(0,255,0),
                    fixe=True,
                    nom="p1")

p2 = Planete(pos=np.array([0,0],dtype=float),
                    vit=np.array([0,0],dtype=float),
                    masse=100,
                    rayon=10,
                    couleur=(0,0,255),
                    fixe=True,
                    nom="p2")

p3 = Planete(pos=np.array([-210,357],dtype=float),
                    vit=np.array([0,0],dtype=float),
                    masse=10,
                    rayon=10,
                    couleur=(255,0,255),
                    fixe=False,
                    nom="p3")

planetes = np.array([p0,p1,p2,p3])

scenar = Scenario(planetes=planetes,
                  long=1000,
                  haut=1000,
                  G=1,
                  t_max=10000,
                  dt=1,
                  res=1e-4,
                  i_sat=3,
                  nom="trois_corps"
                  )