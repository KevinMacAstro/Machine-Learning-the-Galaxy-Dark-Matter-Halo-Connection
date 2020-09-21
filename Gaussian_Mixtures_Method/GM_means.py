import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime




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
        minC=C[ind[d==d.min()],:][0,:]
        return minC


def k_means(Ki):
        C=data[0:Ki,:]
        for i in range(1,Ki):
                if i < Ki:
                        w=np.zeros(N)
                        for j in range(0,N):
                                if j < N:
                                        w[j]=(dist(data[j,:],phi(data[j,:],C,i),0))**2
                                        prob=w/w.sum()
                                        ind=np.linspace(0,N-1,N).astype(int)
                                        PROB=np.vstack((ind,prob)).T
                                        indd=np.random.choice(PROB[:,0].astype(int),p=PROB[:,1])
                                        C[i,:]=data[indd,:]
        return C


def sig(x):
        f=1/(1+np.exp(-x))
        return f


def f(x,mu,sig,d):
	s=x-mu
	arg=-0.5*s.T*np.linalg.inv(sig)*s
	ff=(2*np.pi)**(-d/2)*1/np.sqrt((np.linalg.norm(np.linalg.det(sig))))*np.exp(arg)
	return ff	



def cost_1(C,D,K):
        C_ii=C.reshape(K,D)
        ph=np.zeros(N)
        ind=np.linspace(0,K-1,K).astype(int)
        IND=np.zeros((K,D))
        for i in range(0,D):
                if i < D:
                        IND[:,i]=ind
        for i in range(0,N):
                if i < N:
                        ph[i]=IND[C_ii==phi(data[i,:],C_ii,K)][0]
        ph=ph.astype(int)
        cost=0
        for i in range(0,K):
                cost=cost+sum(dist(data[ph==i],C_ii[i,:],1))
        cost=(1/float(len(data)))*cost
        return cost


def min_cost(K,D):
        l=len(K)
        costs=np.zeros(l)
        for i in range(0,l):
                if i < l:
                        print('Cost-k {} at {}'.format(i,datetime.now().time()))
                        fit=KMeans(n_clusters=K[i]).fit(data)
			C=fit.cluster_centers_
                        costs[i]=cost_1(C,D,K[i])
                        print('Cost= {}'.format(costs[i]))
        return costs

def min_centers(K,D):
        fit=KMeans(n_clusters=K).fit(data)
	C=fit.cluster_centers_
        return C

#galid(0),mstar(1),x1(2),y1(3),z1(4),sfr(5),gband(6),rband(7),metstar(8),metgas(9),snapid(10),x2(11),y2(12),z2(13),m200b(14),a(15),c(16),spin(17),vpeak(18),treeid(19),mvir(20),rvir(21),rs(22),vrms(23),vx(24),vy(25),vz(26),ba(27),ca(28),vmax(29),dist(30)
dat=np.loadtxt('galaxyhalo.dat', delimiter=',',unpack=False)
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

systems=np.vstack((Mvir,vrms,vmax,vpeak,conc,spin,ba,ca,ahalf,Mr,color)).T


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
#cent_DAT=np.delete(cent_DAT,[4,9,11],1)
#all_DAT=np.delete(all_DAT,[4,9,11],1)

# Choice for dataset
print('For the all...')
N=Nall
DATA=systems
TYPE='all1'
D=len(systems[0,:])

standards=np.loadtxt('ILL_{}_standard.dat'.format(TYPE),unpack=False)

data=np.zeros((N,D))
for ni in range(0,D):
        if ni < D:
                data[:,ni]=sig((DATA[:,ni]-standards[ni,0])/standards[ni,1])

kmax=10
k=np.linspace(1,kmax,kmax).astype(int)

print('########################\nStarted Finding means at {}\n############################'.format(datetime.now().time()))
costs=min_cost(k,D)
M=k[np.gradient(np.gradient(costs))==np.gradient(np.gradient(costs)).max()][0]
means1=min_centers((M),D)
means2=min_centers((M+1),D)
means3=min_centers((M+2),D)
means4=min_centers((M+3),D)
means5=min_centers((M+4),D)

np.savetxt('ILLmeans_M_{}.dat'.format(TYPE),means1)
np.savetxt('ILLmeans_Mplus1_{}.dat'.format(TYPE),means2)
np.savetxt('ILLmeans_Mplus2_{}.dat'.format(TYPE),means3)
np.savetxt('ILLmeans_Mplus3_{}.dat'.format(TYPE),means4)
np.savetxt('ILLmeans_Mplus4_{}.dat'.format(TYPE),means5)


print('########################\nFinished Program at {}\n############################'.format(datetime.now().time()))


plt.plot(k,costs)
plt.scatter(M,costs[M-1],s=50,label='M={}'.format(M))
plt.legend(loc='upper right',fontsize=15)
plt.show()





