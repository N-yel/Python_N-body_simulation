import numpy as np
from core import Planete, Scenario

couleur = (0,0,255)
G = 6.674e-11 # en m^3 kg^-1 s^-2

#1 unité de distance = 1 km = 1e3 m
# 1 unité de temps = 1 jours = 3600*24 s
G = G*((3600*24)**2)/((1e3)**3)


soleil = Planete(pos=np.array([0,0,0], dtype=float),
                vit=np.array([0,0,0], dtype=float),
                masse=1988410e24, #kg
                rayon=695700, #km
                couleur=couleur,
                fixe=True,
                nom="soleil")


mercure = Planete(pos=np.array([1.899489215388871E+07, 4.204165905249385E+07,1.693334944158897E+06], dtype=float),
                  vit=np.array([-5.413285046408644E+01, 2.194859967736300E+01, 6.758966993560053E+00], dtype=float)*(3600*24),
                  masse=3.302e23, #kg
                  rayon=2440, #km
                  couleur=couleur,
                  fixe=False,
                  nom="mercure")


venus = Planete(pos=np.array([-9.195385144112816E+07,5.535835164712609E+07,6.048220629802279E+06], dtype=float),
                vit=np.array([-1.820752284769706E+01,-3.017658105262586E+01,6.580832702584960E-01], dtype=float)*(3600*24),
                masse=48.685e23, #kg
                rayon=6051, #km
                couleur=couleur,
                fixe=False,
                nom="venus")

terre = Planete(pos=np.array([-2.546993054069973E+07, 1.448833649541957E+08, -7.309393196061254E+03], dtype=float),
                vit=np.array([-2.981645982255353E+01,-5.280336055653292E+00,-6.389863380211747E-04], dtype=float)*(3600*24),
                masse=5.97219e24, #kg
                rayon=6371, #km
                couleur=couleur,
                fixe=False,
                nom="terre")


mars = Planete(pos=np.array([9.107213936799455E+06, 2.337845503313959E+08,4.676307115335479E+06], dtype=float),
               vit=np.array([-2.329331684632692E+01, 3.001577080095004E+00, 6.342791944424386E-01], dtype=float)*(3600*24),
               masse=6.4171e23,
               rayon=3396.19,
               couleur=couleur,
               fixe=False,
               nom="mars")

jupiter = Planete(pos=np.array([7.237880904345778E+08,1.563651758403323E+08,-1.684298374779323E+07],dtype=float),
                  vit=np.array([-2.914850480846128E+00, 1.339942453189634E+01, 9.575905162939158E-03],dtype=float)*(3600*24),
                  masse=18.9819e26, #kg
                  rayon=71492,
                  couleur=couleur,
                  fixe=False,
                  nom="jupiter")

saturne = Planete(pos=np.array([1.218831716251734E+09, -8.235549677992053E+08, -3.418537285410511E+07], dtype=float),
                  vit=np.array([4.871130208409098E+00, 7.997670501927674E+00, -3.328106808131253E-01], dtype=float)*(3600*24),
                  masse=5.6834e26,
                  rayon=60268,
                  couleur=couleur,
                  fixe=False,
                  nom="saturne")

uranus = Planete(pos=np.array([2.000084411308829E+09, 2.158257760162262E+09,-1.790940860362089E+07], dtype=float),
                 vit=np.array([-5.047858624000558E+00, 4.325458322503441E+00, 8.130965901188003E-02], dtype=float)*(3600*24),
                 masse=86.813e24,
                 rayon=25559,
                 couleur=couleur,
                 fixe=False,
                 nom="uranus")

neptune = Planete(pos=np.array([4.452161647061598E+09,-4.402385494803179E+08,-9.353918371431953E+07], dtype=float),
                  vit=np.array([4.976630999447829E-01, 5.457450965018996E+00,-1.231921649998031E-01], dtype=float)*(3600*24),
                  masse=102.409e24, #kg
                  rayon=24766, #km
                  couleur=couleur,
                  fixe=False,
                  nom="neptune")

y4r = Planete(pos=np.array([-3.038367923974093E+09,-3.365080680705532E+09,1.392624985805583E+08], dtype=float),
                  vit=np.array([3.985381327898765E+00, -3.612277951586504E+00, -1.743214421775274E-02], dtype=float)*(3600*24),
                  masse=1.36E+9, #kg
                  rayon=60, #km
                  couleur=(255,0,0),
                  fixe=False,
                  nom="neptune")


#les planètes sont dans l'ordre alphabetique dans le tableau
planetes = np.array([jupiter,mars,mercure,neptune,saturne,soleil,terre,uranus,venus,y4r])

scenar = Scenario(planetes=planetes,
                  long=1e10,
                  haut=1e10,
                  profond=5e9,
                  G=G,
                  t_max=365*10,
                  i_sat=9,
                  dt=0.1,
                  nom="systeme_solaire"
                  )

