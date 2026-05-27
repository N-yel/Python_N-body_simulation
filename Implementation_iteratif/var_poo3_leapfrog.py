import numpy as np
import models_leapfrog as models

"""Ce fichier sera en 3d"""
long = 1e9
large = 1e9
profond = 1e9

G = 6.674e-11 # en m^3 kg^-1 s^-2

#1 unité de distance = 1 km = 1e3 m
# 1 unité de temps = 10 jours = 3600*24*10 s
G = G*((3600*24*10)**2)/((1e3)**3)



t = 0
t_max = 10000
dt = 1
res = 0

# Pas de satellite
i_sat = 1

soleil = models.Planete(pos=np.array([0,0,0], dtype=float),
                       vit=np.array([0,0,0], dtype=float),
                       acc=np.array([0,0,0], dtype=float),
                       m=1988410e24, #kg
                       r=695700, #km
                       dt=dt,
                       fixe=True)


mercure = models.Planete(pos=np.array([-3.156769979123143e7, 3.747284168204751E+07,5.964978421471700E+06], dtype=float),
                       vit=np.array([-4.709207688822652E+01,-2.939763681694813E+01,1.952870866336399E+00], dtype=float)*(3600*24*10),
                       acc=np.array([0,0,0], dtype=float),
                       m=3.302e23, #kg
                       r=2440, #km
                       dt=dt,
                       fixe=False)


venus = models.Planete(pos=np.array([-9.195385144112816E+07,5.535835164712609E+07,6.048220629802279E+06], dtype=float),
                       vit=np.array([-1.820752284769706E+01,-3.017658105262586E+01,6.580832702584960E-01], dtype=float)*(3600*24*10),
                       acc=np.array([0,0,0], dtype=float),
                       m=48.685e23, #kg
                       r=6051, #km
                       dt=dt,
                       fixe=False)

terre = models.Planete(pos=np.array([-3.366200980958445E+07, 1.431839583709429E+08, 6.381413921842724E+04], dtype=float),
                       vit=np.array([-2.949017064804366E+01, -6.935410623488982E+00, -4.959085158884324E-03], dtype=float)*(3600*24*10),
                       acc=np.array([0,0,0], dtype=float),
                       m=5.97219e24, #kg
                       r=6371, #km
                       dt=dt,
                       fixe=False)


mars = models.Planete(pos=np.array([-1.639995573338535E+08,-1.659634274721135E+08, 6.367700048009604E+05], dtype=float),
                      vit=np.array([1.817929531528863E+01, -1.497589661087009E+01,-7.672790461923418E-01], dtype=float)*(3600*24*10),
                      acc=np.array([0,0,0], dtype=float),
                      m=6.4171e23,
                      r=3396.19,
                      dt=dt,
                      fixe=False)

jupiter = models.Planete(pos=np.array([-4.413622796813691E+06,7.677737470134093E+08,-2.960560464215636E+06],dtype=float),
                    vit=np.array([-1.322811417354989E+01, 5.354176648271677E-01, 2.952118180889938E-01],dtype=float)*(3600*24*10),
                    acc=np.array([0,0,0], dtype=float),
                    m=18.9819e26, #kg
                    r=71492,
                    dt=dt,
                    fixe=False)

saturne = models.Planete(pos=np.array([-8.506298317861040E+08,1.063640738759498E+09,1.469390189216477E+07], dtype=float),
                       vit=np.array([-8.090063837841326E+00, -6.057674952079847E+00, 4.277267263809683E-01], dtype=float)*(3600*24*10),
                       acc=np.array([0,0,0], dtype=float),
                       m=5.6834e26,
                       r=60268,
                       dt=dt,
                       fixe=False)

uranus = models.Planete(pos=np.array([-2.733326091130086E+09, 1.468654611616128E+08, 3.620464174468803E+07], dtype=float),
                       vit=np.array([-4.343335292024516E-01, -7.117610086927938E+00, -2.114970279868400E-02], dtype=float)*(3600*24*10),
                       acc=np.array([0,0,0], dtype=float),
                       m=86.813e24,
                       r=25559,
                       dt=dt,
                       fixe=False)

neptune = models.Planete(pos=np.array([-3.038367923974093E+09,-3.365080680705532E+09,1.392624985805583E+08], dtype=float),
                       vit=np.array([3.985381327898765E+00, -3.612277951586504E+00, -1.743214421775274E-02], dtype=float)*(3600*24*10),
                       acc=np.array([0,0,0], dtype=float),
                       m=102.409e24, #kg
                       r=24766, #km
                       dt=dt,
                       fixe=False)


planetes = np.array([soleil,mercure,venus,terre,mars,jupiter,saturne,uranus,neptune])
