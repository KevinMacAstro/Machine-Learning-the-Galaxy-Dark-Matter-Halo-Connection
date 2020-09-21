import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sc
import scipy as sp
import scipy.ndimage as scifilt
import scipy.stats as scistat
import scipy.optimize as opt
from sklearn.mixture import GaussianMixture

from datetime import datetime



####### Global Definitions ########
def sig(x):
        f=1/(1+np.exp(-x))
        return f

def tanh(x):
        f=2*sig(2*x)-1
        return f

def gauss(V,mu,h):
	N=len(mu)
	D=len(V)
	d=np.zeros((N,D))
	for j in range(0,D):
		if j < D:
			d[:,j]=(V[j]-mu[:,j])**2
	S=-np.sum(d,axis=1)/h**2
	f=np.exp(S)
	F=((2*np.pi)**D*N*h**D)**(-1)*np.sum(f)
	return F

def bandwidth(h):
	N=np.floor(len(data)/100.).astype(int)
	F=np.zeros((N))
	datarand=np.floor(np.random.rand(N)*len(data)).astype(int)
	for i in range(0,N):
		if i < N:
			V=data[datarand[i],:]
			mu=np.delete(data,datarand[i],0)
			F[i]=np.log(gauss(V,mu,h))
	
	return -np.sum(F)/N


print('########################\nStarted Program at {}\n############################'.format(datetime.now().time()))
############# Main ##################
#galid(0),mstar(1),x1(2),y1(3),z1(4),sfr(5),gband(6),rband(7),metstar(8),metgas(9),snapid(10),x2(11),y2(12),z2(13),m200b(14),a(15),c(16),spin(17),vpeak(18),treeid(19),mvir(20),rvir(21),rs(22),vrms(23),vx(24),vy(25),vz(26),ba(27),ca(28),vmax(29),dist(30)
dat=np.loadtxt('centralhalo.dat', delimiter=',',unpack=False)
mstar=np.log10(dat[:,1])
sfr=dat[:,5]
Mr=dat[:,7]
color=dat[:,6]-dat[:,7]
M200b=np.log10(dat[:,14])
ahalf=dat[:,15]
conc=dat[:,16]
spin=dat[:,17]
vpeak=np.log10(dat[:,18])
Mvir=np.log10(dat[:,20])
vrms=np.log10(dat[:,23])
vmax=np.log10(dat[:,29])
ba=dat[:,27]
ca=dat[:,28]

systems=np.vstack((Mvir,vrms,vmax,vpeak,conc,ahalf,Mr)).T

#DAT=np.loadtxt('Hw13_cat.dat',unpack=False)
Nall=len(systems)
#DAT[:,7]=np.log10(DAT[:,7])
#DAT[:,11]=np.log10(DAT[:,11])
#DAT[:,17]=np.log10(DAT[:,17])
#DAT[:,18]=np.log10(DAT[:,18])

#sat_dat=DAT[DAT[:,12]!=-1]
#Nsat=len(sat_dat)
#cent_dat=DAT[DAT[:,12]==-1]
#Ncent=len(cent_dat)

#sat_DAT=np.delete(sat_dat,[0,1,2,3,4,5,6,12],1)
#cent_DAT=np.delete(cent_dat,[0,1,2,3,4,5,6,12],1)
#all_DAT=np.delete(DAT,[0,1,2,3,4,5,6,12],1)

# Choice for dataset
N=Nall
DATA=systems
TYPE='cent'

print('For {}...'.format(TYPE))

### Start calculation
D=len(systems[0,:])
data=np.zeros((N,D))
standard=np.zeros((D,2))
for ni in range(0,D):
	if ni < D:
		standard[ni,0]=np.mean(DATA[:,ni])
		standard[ni,1]=np.std(DATA[:,ni])
        	data[:,ni]=tanh((DATA[:,ni]-standard[ni,0])/standard[ni,1])
np.savetxt('ILL_{}_standard.dat'.format(TYPE),standard)
print('########################\nPrinted Standard at {}\n############################'.format(datetime.now().time()))



NN=20
h_i=np.linspace(0.02,0.3,NN)
h_l=np.zeros(NN)
for i in range(0,NN):
	if i < NN:
		h_l[i]=bandwidth(h_i[i])

binwidth=h_i[h_l==h_l.min()]
np.savetxt('ILL_{}_binwidth.dat'.format(TYPE),binwidth)
print('########################\nPrinted binwidth at {}\n############################'.format(datetime.now().time()))


mi=0
ma=1
axis=np.arange(mi,ma+binwidth,binwidth)
axis[-1]=np.floor(axis[-1])

data_digits=np.digitize(data,axis,right=True)

Nbin=len(axis)
diag1=np.bincount(data_digits[:,0],minlength=Nbin)
diag2=np.bincount(data_digits[:,1],minlength=Nbin)
diag3=np.bincount(data_digits[:,2],minlength=Nbin)
diag4=np.bincount(data_digits[:,3],minlength=Nbin)
diag5=np.bincount(data_digits[:,4],minlength=Nbin)
diag6=np.bincount(data_digits[:,5],minlength=Nbin)
diag7=np.bincount(data_digits[:,6],minlength=Nbin)
diag8=np.bincount(data_digits[:,7],minlength=Nbin)
diag9=np.bincount(data_digits[:,8],minlength=Nbin)
diag10=np.bincount(data_digits[:,9],minlength=Nbin)
diag11=np.bincount(data_digits[:,10],minlength=Nbin)
#diag12=np.bincount(data_digits[:,11],minlength=Nbin)

diagtruth=np.zeros((12,Nbin))
diagtruth[0,:]=diag1!=0
diagtruth[1,:]=diag2!=0
diagtruth[2,:]=diag3!=0
diagtruth[3,:]=diag4!=0
diagtruth[4,:]=diag5!=0
diagtruth[5,:]=diag6!=0
diagtruth[6,:]=diag7!=0
diagtruth[7,:]=diag8!=0
diagtruth[8,:]=diag9!=0
diagtruth[9,:]=diag10!=0
diagtruth[10,:]=diag11!=0
#diagtruth[11,:]=diag12!=0

np.savetxt('ILL_{}_truth.dat'.format(TYPE),diagtruth)

Diag1=diag1[diagtruth[0,:]==1]
Diag2=diag2[diagtruth[1,:]==1]
Diag3=diag3[diagtruth[2,:]==1]
Diag4=diag4[diagtruth[3,:]==1]
Diag5=diag5[diagtruth[4,:]==1]
Diag6=diag6[diagtruth[5,:]==1]
Diag7=diag7[diagtruth[6,:]==1]
Diag8=diag8[diagtruth[7,:]==1]
Diag9=diag9[diagtruth[8,:]==1]
Diag10=diag10[diagtruth[9,:]==1]
Diag11=diag11[diagtruth[10,:]==1]
#Diag12=diag12[diagtruth[11,:]==1]

data_diag=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag5,Diag6,Diag7,Diag8,Diag9,Diag10,Diag11))

np.savetxt('ILL_{}_diag.dat'.format(TYPE),data_diag)
print('########################\nPrinted Observ. Diag at {}\n############################'.format(datetime.now().time()))



print('########################\nStart Boot Subsampling at {}\n############################'.format(datetime.now().time()))
L=len(data_diag)
Nboot=N
DIAG_boot=np.zeros((Nboot,L))
for i in range(0,Nboot):
	if i < Nboot:
		print('Boot number {}/{}.'.format(i,Nboot))
		boot=data[np.floor(np.random.rand(N)*N).astype(int),:]
		boot_digits=np.digitize(boot,axis,right=True)
		diag_boot1=np.bincount(boot_digits[:,0],minlength=Nbin)
		diag_boot2=np.bincount(boot_digits[:,1],minlength=Nbin)
                diag_boot3=np.bincount(boot_digits[:,2],minlength=Nbin)
                diag_boot4=np.bincount(boot_digits[:,3],minlength=Nbin)
                diag_boot5=np.bincount(boot_digits[:,4],minlength=Nbin)
                diag_boot6=np.bincount(boot_digits[:,5],minlength=Nbin)
                diag_boot7=np.bincount(boot_digits[:,6],minlength=Nbin)
                diag_boot8=np.bincount(boot_digits[:,7],minlength=Nbin)
                diag_boot9=np.bincount(boot_digits[:,8],minlength=Nbin)
                diag_boot10=np.bincount(boot_digits[:,9],minlength=Nbin)
                diag_boot11=np.bincount(boot_digits[:,10],minlength=Nbin)
                #diag_boot12=np.bincount(boot_digits[:,11],minlength=Nbin)

		Diag_boot1=diag_boot1[diagtruth[0,:]==1]
		Diag_boot2=diag_boot2[diagtruth[1,:]==1]
		Diag_boot3=diag_boot3[diagtruth[2,:]==1]
		Diag_boot4=diag_boot4[diagtruth[3,:]==1]
		Diag_boot5=diag_boot5[diagtruth[4,:]==1]
		Diag_boot6=diag_boot6[diagtruth[5,:]==1]
		Diag_boot7=diag_boot7[diagtruth[6,:]==1]
		Diag_boot8=diag_boot8[diagtruth[7,:]==1]
		Diag_boot9=diag_boot9[diagtruth[8,:]==1]
		Diag_boot10=diag_boot10[diagtruth[9,:]==1]
		Diag_boot11=diag_boot11[diagtruth[10,:]==1]
		#Diag_boot12=diag_boot12[diagtruth[11,:]==1]

		DIAG_boot[i,:]=np.hstack((Diag_boot1,Diag_boot2,Diag_boot3,Diag_boot4,Diag_boot5,Diag_boot6,Diag_boot7,Diag_boot8,Diag_boot9,Diag_boot10,Diag_boot11))


print('########################\nStart Boot Covariance Calc at {}\n############################'.format(datetime.now().time()))			
DIAG_avg=np.average(DIAG_boot,axis=0)
boot_cov=np.zeros((L,L))
for i in range(0,L):
	if i < L:
		print('{} of {}'.format(i,L))
		for j in range(0,L):	
			if j < L:
				for k in range(0,Nboot):
					if k < Nboot:
						boot_cov[i,j]=boot_cov[i,j]+(DIAG_boot[k,j]-DIAG_avg[j])*(DIAG_boot[k,i]-DIAG_avg[i])
				

BOOT_cov=boot_cov/Nboot
np.savetxt('ILL_{}_cov.dat'.format(TYPE),BOOT_cov)
print('########################\nPrinted Observ. Cov\n############################')
print('########################\nProgram Complete {} \n############################'.format(datetime.now().time()))

