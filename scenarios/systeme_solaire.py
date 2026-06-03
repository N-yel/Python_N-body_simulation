import numpy as np
from core import Planete, Scenario

couleur = (0,0,255)
G = 6.674e-11 # en m^3 kg^-1 s^-2

#1 unité de distance = 1 km = 1e3 m
# 1 unité de temps = 10 jours = 3600*24*10 s
G = G*((3600*24*10)**2)/((1e3)**3)


soleil = Planete(pos=np.array([0,0,0], dtype=float),
                vit=np.array([0,0,0], dtype=float),
                masse=1988410e24, #kg
                rayon=695700, #km
                couleur=couleur,
                fixe=True,
                nom="soleil")


mercure = Planete(pos=np.array([-3.156769979123143e7, 3.747284168204751E+07,5.964978421471700E+06], dtype=float),
                  vit=np.array([-4.709207688822652E+01,-2.939763681694813E+01,1.952870866336399E+00], dtype=float)*(3600*24*10),
                  masse=3.302e23, #kg
                  rayon=2440, #km
                  couleur=couleur,
                  fixe=False,
                  nom="mercure")


venus = Planete(pos=np.array([-9.195385144112816E+07,5.535835164712609E+07,6.048220629802279E+06], dtype=float),
                vit=np.array([-1.820752284769706E+01,-3.017658105262586E+01,6.580832702584960E-01], dtype=float)*(3600*24*10),
                masse=48.685e23, #kg
                rayon=6051, #km
                couleur=couleur,
                fixe=False,
                nom="venus")

terre = Planete(pos=np.array([-3.366200980958445E+07, 1.431839583709429E+08, 6.381413921842724E+04], dtype=float),
                vit=np.array([-2.949017064804366E+01, -6.935410623488982E+00, -4.959085158884324E-03], dtype=float)*(3600*24*10),
                masse=5.97219e24, #kg
                rayon=6371, #km
                couleur=couleur,
                fixe=False,
                nom="terre")


mars = Planete(pos=np.array([-1.639995573338535E+08,-1.659634274721135E+08, 6.367700048009604E+05], dtype=float),
               vit=np.array([1.817929531528863E+01, -1.497589661087009E+01,-7.672790461923418E-01], dtype=float)*(3600*24*10),
               masse=6.4171e23,
               rayon=3396.19,
               couleur=couleur,
               fixe=False,
               nom="mars")

jupiter = Planete(pos=np.array([-4.413622796813691E+06,7.677737470134093E+08,-2.960560464215636E+06],dtype=float),
                  vit=np.array([-1.322811417354989E+01, 5.354176648271677E-01, 2.952118180889938E-01],dtype=float)*(3600*24*10),
                  masse=18.9819e26, #kg
                  rayon=71492,
                  couleur=couleur,
                  fixe=False,
                  nom="jupiter")

saturne = Planete(pos=np.array([-8.506298317861040E+08,1.063640738759498E+09,1.469390189216477E+07], dtype=float),
                  vit=np.array([-8.090063837841326E+00, -6.057674952079847E+00, 4.277267263809683E-01], dtype=float)*(3600*24*10),
                  masse=5.6834e26,
                  rayon=60268,
                  couleur=couleur,
                  fixe=False,
                  nom="saturne")

uranus = Planete(pos=np.array([-2.733326091130086E+09, 1.468654611616128E+08, 3.620464174468803E+07], dtype=float),
                 vit=np.array([-4.343335292024516E-01, -7.117610086927938E+00, -2.114970279868400E-02], dtype=float)*(3600*24*10),
                 masse=86.813e24,
                 rayon=25559,
                 couleur=couleur,
                 fixe=False,
                 nom="uranus")

neptune = Planete(pos=np.array([-3.038367923974093E+09,-3.365080680705532E+09,1.392624985805583E+08], dtype=float),
                  vit=np.array([3.985381327898765E+00, -3.612277951586504E+00, -1.743214421775274E-02], dtype=float)*(3600*24*10),
                  masse=102.409e24, #kg
                  rayon=24766, #km
                  couleur=couleur,
                  fixe=False,
                  nom="neptune")

#les planètes sont dans l'ordre alphabetique dans le tableau
planetes = np.array([jupiter,mars,mercure,neptune,saturne,soleil,terre,uranus,venus])

scenar = Scenario(planetes=planetes,
                  long=5e9,
                  haut=5e9,
                  profond=5e9,
                  G=G,
                  t_max=7305, #de sorte a faire 200 ans, un pas de temps etant égal à 10 jours
                  i_sat=-1,
                  dt=0.1,
                  nom="systeme_solaire"
                  )

