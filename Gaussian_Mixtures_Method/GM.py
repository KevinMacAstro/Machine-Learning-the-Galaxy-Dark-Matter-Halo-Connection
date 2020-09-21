import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from scipy.stats import multivariate_normal
import scipy.stats as stat

def dist(x,y,t):
        # t==0 is for when x and y are only 1 point.
        if t==0:
                d=np.sqrt(sum((x-y)**2))
        else:
                d=np.sqrt(np.sum((x-y)**2,axis=1))
        return d

def phi(x,C,I):
        ind=np.linspace(0,I-1,I).astype(int)
        d=np.zeros(I)
        for j in range(0,I):
                if j < I:
                        d[j]=dist(x,C[j,:],0)
        minC=ind[d==d.min()][0]
        return minC

def sig(x):
        f=1/(1+np.exp(-x))
        return f


def f(x,Mu,Sig,d):
        s=x-Mu
	N=len(s)
	arg=-0.5*np.sum(np.tensordot(s,np.dot(np.linalg.inv(Sig),s.T),axes=1),axis=0)
        ff=(2*np.pi)**(-d/2)*1/np.sqrt((np.linalg.norm(np.linalg.det(Sig))))*np.exp(arg)
        return ff


def f_params(K,weight,data):
	dim=len(data[0,:])
	N=len(weight[0,:])
	WM=np.zeros(K)
	for i in range(0,K):
        	if i < K:
                	for j in range(0,N):
                        	if j < N:
                                	WM[i]=WM[i]+weight[i,j]
	
	mu=np.zeros((K,dim))
	for i in range(0,K):
        	if i < K:
                	for j in range(0,N):
                        	if j < N:
                                	mu[i,:]=mu[i,:]+weight[i,j]*data[j,:]
                	mu[i,:]=mu[i,:]/WM[i]
	
	SigM=np.zeros((K,dim,dim))
	for k in range(0,K):
        	if k < K:
			for d1 in range(0,dim):
				if d1 < dim: 
					for d2 in range(0,dim):
						if d2 < dim:
                					SigM[k,d1,d2]=np.sum((weight[k,:]/WM[k])*(data[:,d1]-mu[k,d1])*(data[:,d2]-mu[k,d2]))
	return mu,SigM

def f_dist(K,mean,covar,data):
	N=len(data)
	f=np.zeros((K,N))
	#Eigendecomp to prevent singular covar matrices		
	if np.any(np.linalg.det(covar)==0.0):
		for j in range(0,K):
			if j < K:
				l,v=np.linalg.eig(covar[j,:])
				l[l<0]=0.0
				l=np.diag(l)
				covar[j,:]=np.real(np.dot(v,np.dot(l,np.linalg.inv(v)))).T
        for j in range(0,K):
                if j < K:
                        var=multivariate_normal(mean=mean[j,:],cov=covar[j,:])
                        f[j,:]=var.pdf(data)
	return f


def update_weight(K,mean,covar,data):
	N=len(data)
	weight=np.zeros((K,N))
	f=f_dist(K,mean,covar,data)
	for j in range(0,K):
        	if j < K:
                	weight[j,:]=f[j,:]/np.sum(f,axis=0)
	return weight,f

def cost(weight,f):
	s=weight*f
	s[s==0.0]=0.000000000001
	cost=np.sum(-np.log10(s))
	return cost

def GMM_points(weight,mean,covar):
	dim=len(mean[0,:])
	N=len(weight[0,:])
	W=np.ceil(np.sum(weight,axis=1)).astype(int)
	K=len(W)
	points=[]
	for i in range(0,K):
		if i < K:
			points.append(np.random.multivariate_normal(mean[i,:],covar[i,:],size=W[i]))
	points=np.array(points)
	points=np.concatenate(points)
	point=points[0:N,:]
	point[point>1.0]=data.max()
	point[point<0.0]=data.min()
	return point

def model_chi2(model_diag,obs_diag,obs_cov):
	s=model_diag-obs_diag
	obsinv=np.linalg.inv(obs_cov)
	cost=np.dot(s,np.dot(obs_cov,s))
	return cost




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

systems=np.vstack((M200b,Mvir,vrms,vmax,vpeak,conc,spin,ba,ca,ahalf,Mr,color)).T

#DAT=np.loadtxt('Hw13_cat.dat',unpack=False)

Nall=len(dat)
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
#all_DAT=np.delete(all_DAT,[4,9,11],1)
#cent_DAT=np.delete(cent_DAT,[4,9,11],1)


# Choice for dataset
print('For the all...')
N=Nall
DATA=systems
TYPE='cent'
D=len(systems[0,:])

mean,std=np.loadtxt('ILL_{}_standard.dat'.format(TYPE),unpack=True)

data=np.zeros((N,D))
for ni in range(0,D):
        if ni < D:
                data[:,ni]=sig((DATA[:,ni]-mean[ni])/std[ni])


M=np.loadtxt('ILLmeans_M_{}.dat'.format(TYPE),unpack=False)
M1=np.loadtxt('ILLmeans_Mplus1_{}.dat'.format(TYPE),unpack=False)
M2=np.loadtxt('ILLmeans_Mplus2_{}.dat'.format(TYPE),unpack=False)
M3=np.loadtxt('ILLmeans_Mplus3_{}.dat'.format(TYPE),unpack=False)
M4=np.loadtxt('ILLmeans_Mplus4_{}.dat'.format(TYPE),unpack=False)
NM=len(M[:,0])
NM1=len(M1[:,0])
NM2=len(M2[:,0])
NM3=len(M3[:,0])
NM4=len(M4[:,0])


# Initilize weights.
wM=np.zeros((NM,N))
wM1=np.zeros((NM1,N))
wM2=np.zeros((NM2,N))
wM3=np.zeros((NM3,N))
wM4=np.zeros((NM4,N))
for i in range(0,N):
	if i < N:
		ind=phi(data[i,:],M,NM)
		wM[ind,i]=1
for i in range(0,N):
        if i < N:
                ind=phi(data[i,:],M1,NM1)
                wM1[ind,i]=1
for i in range(0,N):
        if i < N:
                ind=phi(data[i,:],M2,NM2)
                wM2[ind,i]=1
for i in range(0,N):
        if i < N:
                ind=phi(data[i,:],M3,NM3)
                wM3[ind,i]=1
for i in range(0,N):
        if i < N:
                ind=phi(data[i,:],M4,NM4)
                wM4[ind,i]=1


cost_old=np.exp(500)
Mu,Sig=f_params(NM,wM,data)
w,f=update_weight(NM,Mu,Sig,data)
cost_new=cost(w,f)
its=1
while abs(cost_new-cost_old)/cost_old > 0.0000001:
	print('Iteration {} for M'.format(its))
	Mu,Sig=f_params(NM,w,data)
	w,f=update_weight(NM,Mu,Sig,data)
	cost_old=cost_new
	cost_new=cost(w,f)
	its=its+1
Mu,Sig=f_params(NM,w,data)	

predict_data=GMM_points(w,Mu,Sig)

predict_data[predict_data==predict_data.min()]=data.min()
predict_data[predict_data==predict_data.max()]=data.max()


cost_old=np.exp(500)
Mu,Sig=f_params(NM1,wM1,data)
w,f=update_weight(NM1,Mu,Sig,data)
cost_new=cost(w,f)
its=1
while abs(cost_new-cost_old)/cost_old > 0.0000001:
        print('Iteration {} for M1'.format(its))
        Mu,Sig=f_params(NM1,w,data)
        w,f=update_weight(NM1,Mu,Sig,data)
        cost_old=cost_new
        cost_new=cost(w,f)
        its=its+1
Mu,Sig=f_params(NM1,w,data)

predict_data1=GMM_points(w,Mu,Sig)

predict_data1[predict_data1==predict_data1.min()]=data.min()
predict_data1[predict_data1==predict_data1.max()]=data.max()

cost_old=np.exp(500)
Mu,Sig=f_params(NM2,wM2,data)
w,f=update_weight(NM2,Mu,Sig,data)
cost_new=cost(w,f)
its=1
while abs(cost_new-cost_old)/cost_old > 0.0000001:
        print('Iteration {} for M2'.format(its))
        Mu,Sig=f_params(NM2,w,data)
        w,f=update_weight(NM2,Mu,Sig,data)
        cost_old=cost_new
        cost_new=cost(w,f)
        its=its+1
Mu,Sig=f_params(NM2,w,data)

predict_data2=GMM_points(w,Mu,Sig)

predict_data2[predict_data2==predict_data2.min()]=data.min()
predict_data2[predict_data2==predict_data2.max()]=data.max()

cost_old=np.exp(500)
Mu,Sig=f_params(NM3,wM3,data)
w,f=update_weight(NM3,Mu,Sig,data)
cost_new=cost(w,f)
its=1
while abs(cost_new-cost_old)/cost_old > 0.0000001:
        print('Iteration {} for M3'.format(its))
        Mu,Sig=f_params(NM3,w,data)
        w,f=update_weight(NM3,Mu,Sig,data)
        cost_old=cost_new
        cost_new=cost(w,f)
        its=its+1
Mu,Sig=f_params(NM3,w,data)

predict_data3=GMM_points(w,Mu,Sig)

predict_data3[predict_data3==predict_data3.min()]=data.min()
predict_data3[predict_data3==predict_data3.max()]=data.max()

cost_old=np.exp(500)
Mu,Sig=f_params(NM4,wM4,data)
w,f=update_weight(NM4,Mu,Sig,data)
cost_new=cost(w,f)
its=1
while abs(cost_new-cost_old)/cost_old > 0.0000001:
        print('Iteration {} for M4'.format(its))
        Mu,Sig=f_params(NM4,w,data)
        w,f=update_weight(NM4,Mu,Sig,data)
        cost_old=cost_new
        cost_new=cost(w,f)
        its=its+1
Mu,Sig=f_params(NM4,w,data)

predict_data4=GMM_points(w,Mu,Sig)

predict_data4[predict_data4==predict_data4.min()]=data.min()
predict_data4[predict_data4==predict_data4.max()]=data.max()


np.savetxt('ILL_{}_predictdata.dat'.format(TYPE),predict_data)
np.savetxt('ILL_{}_predictdata1.dat'.format(TYPE),predict_data1)
np.savetxt('ILL_{}_predictdata2.dat'.format(TYPE),predict_data2)
np.savetxt('ILL_{}_predictdata3.dat'.format(TYPE),predict_data3)
np.savetxt('ILL_{}_predictdata4.dat'.format(TYPE),predict_data4)


observe_diag=np.loadtxt('ILL_{}_diag.dat'.format(TYPE),unpack=False).astype(int)
observe_cov=np.loadtxt('ILL_{}_cov.dat'.format(TYPE),unpack=False)


#if TYPE=='cent' or TYPE=='all':

	# HW13 All
        #a=np.arange(105,132,1)
        #b=np.arange(222,251,1)
	# Shuffled cent
	#a=np.arange(107,136,1)
	#b=np.arange(236,265,1)
	# HW13 cent
	#a=np.arange(180,230,1)
	#b=np.arange(387,437,1)
	#c=np.hstack((a,b))
	#observe_diag=np.delete(observe_diag,c)
        # HW13 All
        #observe_diag=observe_diag[:-26]
	# Shuffled cent
        #observe_diag=observe_diag[:-25]
	# HW13 cent
	#observe_diag=observe_diag[:-41]
        # HW13 All
        #observe_cov=observe_cov[:-26,:-26]
	# Shuffled Cent
	#observe_cov=observe_cov[:-25,:-25]
	# HW13 Cent
	#observe_cov=observe_cov[:-41,:-41]
	#observe_cov=np.delete(observe_cov,c,axis=1)
	#observe_cov=np.delete(observe_cov,c,axis=0)

#l,v=np.linalg.eig(observe_cov)
#while np.any(l<0.0):
#	l[l<0]=0.0
#	l=np.diag(l)
#	obs_cov=np.real(np.dot(v,np.dot(l,np.linalg.inv(v)))).T
#	l,v=np.linalg.eig(obs_cov)

diagtruth=np.loadtxt('ILL_{}_truth.dat'.format(TYPE),unpack=False)
binwidth=np.loadtxt('ILL_{}_binwidth.dat'.format(TYPE),unpack=False)
mi=0
ma=1
axis=np.arange(mi,ma+binwidth,binwidth)
axis[-1]=np.floor(axis[-1])

predict_digits=np.digitize(predict_data,axis,right=True)

Nbin=len(axis)
diag1=np.bincount(predict_digits[:,0],minlength=Nbin)
diag2=np.bincount(predict_digits[:,1],minlength=Nbin)
diag3=np.bincount(predict_digits[:,2],minlength=Nbin)
diag4=np.bincount(predict_digits[:,3],minlength=Nbin)
if TYPE=='cent' or TYPE=='all':
	diag5=np.bincount(predict_digits[:,4],minlength=Nbin)
	diag6=np.bincount(predict_digits[:,5],minlength=Nbin)
	diag7=np.bincount(predict_digits[:,6],minlength=Nbin)
	diag8=np.bincount(predict_digits[:,7],minlength=Nbin)
	diag9=np.bincount(predict_digits[:,8],minlength=Nbin)
	diag10=np.bincount(predict_digits[:,9],minlength=Nbin)
	diag11=np.bincount(predict_digits[:,10],minlength=Nbin)
	diag12=np.bincount(predict_digits[:,11],minlength=Nbin)
if TYPE=='sat' or TYPE=='al':
        diag6=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,8],minlength=Nbin)

Diag1=diag1[diagtruth[0,:]==1]
Diag2=diag2[diagtruth[1,:]==1]
Diag3=diag3[diagtruth[2,:]==1]
Diag4=diag4[diagtruth[3,:]==1]
if TYPE=='cent' or TYPE=='all':
	Diag5=diag5[diagtruth[4,:]==1]
	Diag6=diag6[diagtruth[5,:]==1]
	Diag7=diag7[diagtruth[6,:]==1]
	Diag8=diag8[diagtruth[7,:]==1]
	Diag9=diag9[diagtruth[8,:]==1]
	Diag10=diag10[diagtruth[9,:]==1]
	Diag11=diag11[diagtruth[10,:]==1]
	Diag12=diag12[diagtruth[11,:]==1]
if TYPE=='sat' or TYPE=='al':
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]
  
if TYPE=='cent' or TYPE=='all':
	predict_diag=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag5,Diag6,Diag7,Diag8,Diag9,Diag10,Diag11,Diag12))
if TYPE=='sat' or TYPE=='al':
	predict_diag=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag6,Diag7,Diag8,Diag9,Diag11))

predict_digits=np.digitize(predict_data1,axis,right=True)

Nbin=len(axis)
diag1=np.bincount(predict_digits[:,0],minlength=Nbin)
diag2=np.bincount(predict_digits[:,1],minlength=Nbin)
diag3=np.bincount(predict_digits[:,2],minlength=Nbin)
diag4=np.bincount(predict_digits[:,3],minlength=Nbin)
if TYPE=='cent' or TYPE=='all':
        diag5=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag6=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,8],minlength=Nbin)
        diag10=np.bincount(predict_digits[:,9],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,10],minlength=Nbin)
        diag12=np.bincount(predict_digits[:,11],minlength=Nbin)
if TYPE=='sat' or TYPE=='al':
        diag6=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,8],minlength=Nbin)

Diag1=diag1[diagtruth[0,:]==1]
Diag2=diag2[diagtruth[1,:]==1]
Diag3=diag3[diagtruth[2,:]==1]
Diag4=diag4[diagtruth[3,:]==1]
if TYPE=='cent' or TYPE=='all':
        Diag5=diag5[diagtruth[4,:]==1]
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag10=diag10[diagtruth[9,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]
        Diag12=diag12[diagtruth[11,:]==1]
if TYPE=='sat' or TYPE=='al':
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]

if TYPE=='cent' or TYPE=='all':
        predict_diag1=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag5,Diag6,Diag7,Diag8,Diag9,Diag10,Diag11,Diag12))
if TYPE=='sat' or TYPE=='al':
        predict_diag1=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag6,Diag7,Diag8,Diag9,Diag11))

predict_digits=np.digitize(predict_data2,axis,right=True)

Nbin=len(axis)
diag1=np.bincount(predict_digits[:,0],minlength=Nbin)
diag2=np.bincount(predict_digits[:,1],minlength=Nbin)
diag3=np.bincount(predict_digits[:,2],minlength=Nbin)
diag4=np.bincount(predict_digits[:,3],minlength=Nbin)
if TYPE=='cent' or TYPE=='all':
        diag5=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag6=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,8],minlength=Nbin)
        diag10=np.bincount(predict_digits[:,9],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,10],minlength=Nbin)
        diag12=np.bincount(predict_digits[:,11],minlength=Nbin)
if TYPE=='sat' or TYPE=='al':
        diag6=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,8],minlength=Nbin)

Diag1=diag1[diagtruth[0,:]==1]
Diag2=diag2[diagtruth[1,:]==1]
Diag3=diag3[diagtruth[2,:]==1]
Diag4=diag4[diagtruth[3,:]==1]
if TYPE=='cent' or TYPE=='all':
        Diag5=diag5[diagtruth[4,:]==1]
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag10=diag10[diagtruth[9,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]
        Diag12=diag12[diagtruth[11,:]==1]
if TYPE=='sat' or TYPE=='al':
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]

if TYPE=='cent' or TYPE=='all':
        predict_diag2=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag5,Diag6,Diag7,Diag8,Diag9,Diag10,Diag11,Diag12))
if TYPE=='sat' or TYPE=='al':
        predict_diag2=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag6,Diag7,Diag8,Diag9,Diag11))

predict_digits=np.digitize(predict_data3,axis,right=True)

Nbin=len(axis)
diag1=np.bincount(predict_digits[:,0],minlength=Nbin)
diag2=np.bincount(predict_digits[:,1],minlength=Nbin)
diag3=np.bincount(predict_digits[:,2],minlength=Nbin)
diag4=np.bincount(predict_digits[:,3],minlength=Nbin)
if TYPE=='cent' or TYPE=='all':
        diag5=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag6=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,8],minlength=Nbin)
        diag10=np.bincount(predict_digits[:,9],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,10],minlength=Nbin)
        diag12=np.bincount(predict_digits[:,11],minlength=Nbin)
if TYPE=='sat' or TYPE=='al':
        diag6=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,8],minlength=Nbin)

Diag1=diag1[diagtruth[0,:]==1]
Diag2=diag2[diagtruth[1,:]==1]
Diag3=diag3[diagtruth[2,:]==1]
Diag4=diag4[diagtruth[3,:]==1]
if TYPE=='cent' or TYPE=='all':
        Diag5=diag5[diagtruth[4,:]==1]
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag10=diag10[diagtruth[9,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]
        Diag12=diag12[diagtruth[11,:]==1]
if TYPE=='sat' or TYPE=='al':
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]

if TYPE=='cent' or TYPE=='all':
        predict_diag3=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag5,Diag6,Diag7,Diag8,Diag9,Diag10,Diag11,Diag12))
if TYPE=='sat' or TYPE=='al':
        predict_diag3=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag6,Diag7,Diag8,Diag9,Diag11))

predict_digits=np.digitize(predict_data4,axis,right=True)

Nbin=len(axis)
diag1=np.bincount(predict_digits[:,0],minlength=Nbin)
diag2=np.bincount(predict_digits[:,1],minlength=Nbin)
diag3=np.bincount(predict_digits[:,2],minlength=Nbin)
diag4=np.bincount(predict_digits[:,3],minlength=Nbin)
if TYPE=='cent' or TYPE=='all':
        diag5=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag6=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,8],minlength=Nbin)
        diag10=np.bincount(predict_digits[:,9],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,10],minlength=Nbin)
        diag12=np.bincount(predict_digits[:,11],minlength=Nbin)
if TYPE=='sat' or TYPE=='al':
        diag6=np.bincount(predict_digits[:,4],minlength=Nbin)
        diag7=np.bincount(predict_digits[:,5],minlength=Nbin)
        diag8=np.bincount(predict_digits[:,6],minlength=Nbin)
        diag9=np.bincount(predict_digits[:,7],minlength=Nbin)
        diag11=np.bincount(predict_digits[:,8],minlength=Nbin)

Diag1=diag1[diagtruth[0,:]==1]
Diag2=diag2[diagtruth[1,:]==1]
Diag3=diag3[diagtruth[2,:]==1]
Diag4=diag4[diagtruth[3,:]==1]
if TYPE=='cent' or TYPE=='all':
        Diag5=diag5[diagtruth[4,:]==1]
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag10=diag10[diagtruth[9,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]
        Diag12=diag12[diagtruth[11,:]==1]
if TYPE=='sat' or TYPE=='al':
        Diag6=diag6[diagtruth[5,:]==1]
        Diag7=diag7[diagtruth[6,:]==1]
        Diag8=diag8[diagtruth[7,:]==1]
        Diag9=diag9[diagtruth[8,:]==1]
        Diag11=diag11[diagtruth[10,:]==1]

if TYPE=='cent' or TYPE=='all':
        predict_diag4=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag5,Diag6,Diag7,Diag8,Diag9,Diag10,Diag11,Diag12))
if TYPE=='sat' or TYPE=='al':
        predict_diag4=np.hstack((Diag1,Diag2,Diag3,Diag4,Diag6,Diag7,Diag8,Diag9,Diag11))


modelchi2=np.zeros(5)
modelchi2[0]=model_chi2(predict_diag,observe_diag,observe_cov)
modelchi2[1]=model_chi2(predict_diag1,observe_diag,observe_cov)
modelchi2[2]=model_chi2(predict_diag2,observe_diag,observe_cov)
modelchi2[3]=model_chi2(predict_diag3,observe_diag,observe_cov)
modelchi2[4]=model_chi2(predict_diag4,observe_diag,observe_cov)

np.savetxt('ILL_{}_modelchi2.dat'.format(TYPE),modelchi2)
print('########################\nFinished Program at {}\n############################'.format(datetime.now().time()))












