x=1
v=2
dt=3
a=4
a_new = 5
sum_forces = 6
m=7

x += v * dt + (1/2) * a * dt**2
a_new = sum_forces / m
v = v + (1/2) * (a + a_new) * dt
a = a_new