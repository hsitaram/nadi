import yt
from sys import argv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
    

#=======================================
#read Ghia and Ghia solution
#=======================================
infile=open("Ghia_soln_Re100",'r')

y_ghia = np.array([])
v_ghia = np.array([])

for line in infile:
    splt=line.split()
    y_ghia=np.append(y_ghia,float(splt[0]))
    v_ghia=np.append(v_ghia,float(splt[1]))

infile.close()
#=======================================

#=======================================
#read EulerAMR solution
#=======================================
ds=yt.load(argv[1])
slicedir   = argv[2]
wallmovdir = argv[3]

clength   = 1.0
cwidth    = 1.0
cdepth    = 0.1

fieldname="vel"+wallmovdir
res = 100
slicedepth = cdepth/2

slc = ds.slice(slicedir,slicedepth)
frb = slc.to_frb((1,'cm'), res)
y = np.linspace(0,1,res)
fld = np.array(frb[fieldname])[:,int(res/2)]
#=======================================

#=======================================
#Plot solutions
#=======================================
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3))
ax1.plot(y,fld/np.max(fld),'k',label="EulerAMR")
ax1.plot(y_ghia,v_ghia,'r*',label="Ghia et al.,JCP,48,pp 387-411,1982")
ax1.legend(loc="best")

im=ax2.imshow(np.array(frb[fieldname]),origin="lower")
fig.colorbar(im, ax=ax2)

fig.suptitle("Driven cavity with moving wall along "+wallmovdir)
plt.savefig("vel_drivencavity_"+wallmovdir+".png")
#=======================================

