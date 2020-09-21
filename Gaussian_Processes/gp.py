from gapp import dgp
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import scipy.linalg as LA

def sqexp(theta,x1,x2):
	f=theta[0]**2*np.exp(-(x1-x2)**2/(2*theta[1]**2))
	return f

sigma80=0.8111

(X, Y, Sigma) = np.loadtxt("growthhistory.dat", unpack = 'True')
N=len(X)

xmin = 0.0
xmax = 2.5
nstar = 200
initheta = [0.5,0.5]

g = dgp.DGaussianProcess(X, Y, Sigma, cXstar = (xmin, xmax, nstar))
(rec, theta) = g.gp()

Xstar=rec[:,0]

drec=(rec[1,0]-rec[0,0])/2
rec1=rec[:,0]+drec


deltaP=-(1/sigma80)*rec[:,1]/(1+rec[:,0])
deltaPerr=(1/sigma80)*rec[:,2]/(1+rec[:,0])


OBSCOV=np.zeros((nstar,nstar))
for i in range(0,nstar):
	for j in range(0,nstar):
		kstar1=g.covariance_vector(Xstar[i])
		kstar2=g.covariance_vector(Xstar[j])
		v1=LA.solve(g.L,kstar1)
		v2=LA.solve(g.L,kstar2)
		g.covf.x1=Xstar[i]
		g.covf.x2=Xstar[j]
		covstar=g.covf.covfunc()
		OBSCOV[i,j]=covstar-np.dot(np.transpose(v2),v1)

invz=1/(1+Xstar)

deltaz=rec[:,0][1]
delta=np.zeros(nstar-1)
for i in range(1,nstar):
        delta[i-1]=1-(1/sigma80)*np.sum(rec[0:i,1]/(1+rec[0:i,0]))*deltaz

tmperr=np.zeros(nstar-1)
for k in range(0,nstar-1):
	for i in range(0,k+2):
		for j in range(i,k+2):
			if i==j:
				tmperr[k]=tmperr[k]+invz[i]**2*OBSCOV[i,i]
			#else:
			 #       tmperr[k]=tmperr[k]+2*invz[i]*invz[j]*OBSCOV[i,j]

deltaerr=(deltaz/sigma80)*np.sqrt(tmperr)

f=-(1+Xstar[:-1])*(deltaP[:-1]/delta)





r=2
gs_1=plt.GridSpec(22, 20,hspace=0)
gs_2=plt.GridSpec(22, 20,hspace=0)
gs_3=plt.GridSpec(22, 20,hspace=0)
gs_4=plt.GridSpec(22, 20,hspace=0)
ax1=plt.subplot(gs_1[0:10,0:10])
ax2=plt.subplot(gs_2[0:10,10:20])
ax3=plt.subplot(gs_3[12:22,0:10])
ax4=plt.subplot(gs_4[12:22,10:20])

ax1.fill_between(rec[:,0],(rec[:,1]+2*rec[:,2]),(rec[:,1]-2*rec[:,2]),color='lightblue')
ax1.fill_between(rec[:,0],(rec[:,1]+rec[:,2]),(rec[:,1]-rec[:,2]),color='blue')
ax1.errorbar(rec[:,0],rec[:,1],yerr=None,color='darkblue',lw=r)
ax1.errorbar(X,Y,yerr=Sigma,fmt='o',color='r')

ax2.fill_between(rec1[:-1],(delta+2*deltaerr),(delta-2*deltaerr),color='lightblue')
ax2.fill_between(rec1[:-1],(delta+deltaerr),(delta-deltaerr),color='blue')
ax2.errorbar(rec1[:-1],delta,yerr=None,color='darkblue',lw=r)


ax3.fill_between(rec[:,0],(deltaP+2*deltaPerr),(deltaP-2*deltaPerr),color='lightblue')
ax3.fill_between(rec[:,0],(deltaP+deltaPerr),(deltaP-deltaPerr),color='blue')
ax3.errorbar(rec[:,0],deltaP,yerr=None,color='darkblue',lw=r)


ax4.errorbar(rec1[:-1],f,yerr=None,color='darkblue',lw=r)


ax1.set_ylim(0.1,0.7)
ax1.set_xlim(0.0,2.5)
ax1.set_ylabel(r'$f\sigma_8$',fontsize=20)
ax1.set_xlabel(r'$z$',fontsize=20)
ax2.set_ylim(0.35,1.05)
ax2.set_xlim(0.0,2.5)
ax2.set_ylabel(r'$\delta / \delta_0$',fontsize=20)
ax2.set_xlabel(r'$z$',fontsize=20)
ax3.set_ylim(-0.7,0.0)
ax3.set_xlim(0.0,2.5)
ax3.set_ylabel(r'$\delta\prime / \delta_0$',fontsize=20)
ax3.set_xlabel(r'$z$',fontsize=20)
ax4.set_ylim(0.15,1.5)
ax4.set_xlim(0.0,2.5)
ax4.set_ylabel(r'$f$',fontsize=20)
ax4.set_xlabel(r'$z$',fontsize=20)


plt.show()




