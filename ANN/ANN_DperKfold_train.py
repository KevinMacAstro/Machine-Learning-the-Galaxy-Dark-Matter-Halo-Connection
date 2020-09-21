import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
print('Program started at {}.'.format(datetime.now().time()))


def primes(n):
    divisors = [ d for d in range(2,n//2+1) if n % d == 0 ]
    return [ d for d in divisors if \
             all( d % od != 0 for od in divisors if od != d ) ]


def binarySearch(alist, item):
        first = 0
        last = len(alist)-1
        found = False
        while first<=last and not found:
                midpoint = (first + last)//2
                if alist[midpoint] == item:
                    found = True
                else:
                    if item < alist[midpoint]:
                        last = midpoint-1
                    else:
                        first = midpoint+1
        if found:
                return midpoint
        else:
                print('false')

#Import ANN data from previous section
data=np.loadtxt('HW13_params_2.dat', unpack=False)
np.random.shuffle(data)

#More attention to training set selection. Need to explore parameter space.
ND=len(data)


#Inputs for params_1
#Inputs=data[:,0:5]
#IDs=data[:,5:6]
#outputs=data[:,7:9]


#Inputs for params 2
inputs1=data[:,1]
inputs2=data[:,13]
inputs3=data[:,14]
inputs4=data[:,15]
inputs5=data[:,16]
Inputs=np.vstack((inputs1,inputs2,inputs3,inputs4,inputs5)).T

outputs=data[:,17:19]

#Number of input parameters:
NI=len(Inputs[0,:])

# Standardize on only training set
#Choose training sets:
NT=100000
inputs=Inputs[0:NT,:]
out=outputs[0:NT,0]

# Standardize inputs
Ninputs=np.zeros((NT,(NI)))
standard=np.zeros(((NI),2))
for ni in range(0,(NI)):
	if ni < (NI) :
		standard[ni,0]=np.mean(inputs[:,ni])
		standard[ni,1]=np.std(inputs[:,ni])
                Ninputs[:,ni]=(inputs[:,ni]-standard[ni,0])/standard[ni,1]

#np.savetxt('Standard_HW13_params2_{}_9.6Dper.dat'.format(NT),standard)
Nin=Ninputs


#Standardize on entire HW13
#Ninputs=np.zeros((ND,(NI)))
#standard=np.zeros(((NI),2))
#for ni in range(0,(NI)):
#        if ni < (NI) :
#                standard[ni,0]=np.mean(Inputs[:,ni])
#                standard[ni,1]=np.std(Inputs[:,ni])
#                Ninputs[:,ni]=(Inputs[:,ni]-standard[ni,0])/standard[ni,1]
#np.savetxt('Standard_HW13_params1.dat',standard)

#Choose training sets:
#NLH=primes(ND)[-1]
#LHcount=ND/NLH+1
#for lh in range(0,LHbins):
#	if lh < LHbins:
#		for ni in range(0,NI):
#			if ni < NI:
#				tmpPAR=np.sort(Inputs[:,ni])
#				for nlh in range(1,NLH)
#						tmpPAR[(nlh*16)-1]
						
#test=np.vstack((Inputs[:,1],outputs[:,0])).T
#test2=test[test[:,0].argsort()]

#for i in range(0,(ND-1)):
#	if i < (ND-1):
#		if tmpPAR[(i+1)]==tmpPAR[i]:
#			print('{}'.format(i))

				
	#	LHbinmatrix[ni,



#NT=10000
#Nin=Ninputs[0:NT,:]
#out=outputs[0:NT,0]
#
# Number of hidden layers:
K=NI

b=np.zeros(K)
for i in range(0,K):
	if i < K:
		p=len(Nin[:,0][Nin[:,0]>1])/float(NT)
		b[i]=np.log10(p/(1-p))


Wout=2*np.median(out)

#Assign random weights for initial iteration according to  Xavier Glorot, Yoshua Bengio ;Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, PMLR 9:249-256, 2010. 
r1=1/np.sqrt(NI)
w_1=np.zeros((K,NI))
w_2=np.zeros((K))
for k in range(0,K):
	if k < K:
		for ni in range(0,NI):
			if ni < NI:
				w_1[k,ni]=-r1+2*r1*np.random.rand()

r2=1/np.sqrt(K)
for k in range(0,K):
	if k < K:
		w_2[k]=-r2+2*r2*np.random.rand()



# Train ANN w/ k-fold validation
I=500
ksamp=NT/I
NTK=NT-ksamp
E=np.zeros(I)
RMSE=np.zeros(I)
MAE=np.zeros(I)
E_K=np.zeros(I)
RMSE_K=np.zeros(I)
MAE_K=np.zeros(I)
w_1s=np.zeros((25,I))
w_2s=np.zeros((5,I))
bs=np.zeros((5,I))
Wouts=np.zeros(I)
for i in range(0,I):
	if i < I :
		left=NT-(i+1)*ksamp
		right=NT-i*ksamp
		NinTRAIN=Nin[0:left]
		outTRAIN=out[0:left]
		if i > 0:
			NinTRAIN2=Nin[right:]
			NinTRAIN=np.vstack((NinTRAIN,NinTRAIN2))
			outTRAIN2=out[right:]
			outTRAIN=np.hstack((outTRAIN,outTRAIN2))
		NinKFOLD=Nin[left:right]
		outKFOLD=out[left:right]
		eta=0.00001
		s=np.zeros((NTK,K))
		t=np.zeros((NTK,K))
		for ntk in range(0,NTK):
			if ntk < NTK :
				for k in range(0,K):
					if k < K:
						for ni in range(0,NI):
							if ni < NI:
								s[ntk,k]=s[ntk,k]+(w_1[k,ni]*NinTRAIN[ntk,ni])		
						t[ntk,k]=1/(1+np.exp(-(s[ntk,k]+b[k])))
		
		
		y = np.sum(w_2*t, axis=1)	
		t2=1/(1+np.exp(-y))
		q=Wout*t2
		E[i]=0.5*np.sum((q.T-outTRAIN)**2)						
		RMSE[i]=np.sqrt(1/float(NTK)*np.sum(((q.T-outTRAIN)/outTRAIN)**2))
		MAE[i]=np.sum(abs(q.T-outTRAIN))/NTK

                sk=np.zeros((ksamp,K))
                tk=np.zeros((ksamp,K))
                for ks in range(0,ksamp):
                        if ks < ksamp :
                                for k in range(0,K):
                                        if k < K:
                                                for ni in range(0,NI):
                                                        if ni < NI:
                                                                sk[ks,k]=sk[ks,k]+(w_1[k,ni]*NinKFOLD[ks,ni])
                                                tk[ks,k]=1/(1+np.exp(-(sk[ks,k]+b[k])))


                yk = np.sum(w_2*tk, axis=1)
                tk2=1/(1+np.exp(-yk))
                qk=Wout*tk2

                E_K[i]=0.5*np.sum((qk.T-outKFOLD)**2)
                RMSE_K[i]=np.sqrt(1/float(ksamp)*np.sum(((qk.T-outKFOLD)/outKFOLD)**2))
                MAE_K[i]=np.sum(abs(qk.T-outKFOLD))/ksamp

		W_1=w_1.flatten()
		w_1s[:,i]=W_1
		w_2s[:,i]=w_2
		bs[:,i]=b
		Wouts[i]=Wout


		#Update weights
		del_w_1=np.zeros((K,NI))
		del_w_2=np.zeros((K))
		del_b=np.zeros((K))
		del_Wout=0.
		for k in range(0,K):
			if k < K:
				for ntk in range(0,NTK):
					if ntk < NTK :
						del_w_2[k]=del_w_2[k]-eta*(q[ntk]-outTRAIN[ntk])*t2[ntk]*(1-t2[ntk])*Wout*t[ntk,k]

					
		for k in range(0,K):
			if k < K :
				for ni in range(0,NI):
					if ni < NI :
						for ntk in range(0,NTK):
							if ntk < NTK :
								del_w_1[k,ni]=del_w_1[k,ni]-eta*(q[ntk]-outTRAIN[ntk])*Wout*t2[ntk]*(1-t2[ntk])*w_2[k]*t[ntk,k]*(1-t[ntk,k])*NinTRAIN[ntk,ni]

		for k in range(0,K):
			if k < K:
				for ntk in range(0,NTK):
					if ntk < NTK:
						del_b[k]=del_b[k]-eta*(q[ntk]-outTRAIN[ntk])*t2[ntk]*(1-t2[ntk])*Wout*w_2[k]*t[ntk,k]*(1-t[ntk,k])
		for ntk in range(0,NTK):
			if ntk < NTK:
				del_Wout=del_Wout-eta*(q[ntk]-outTRAIN[ntk])*t2[ntk]
		w_1=w_1+del_w_1
		w_2=w_2+del_w_2
		b=b+del_b
		Wout=Wout+del_Wout

#outs=np.vstack((q,out)).T
#np.savetxt('ANNDper_output_HW13_params2_{}s.{}.dat'.format(NT,I),outs)
#err_outs=np.vstack((E,RMSE,MAE)).T 
#Wouts=np.array([Wout])
#np.savetxt('ANNDper_errors_HW13_params2_{}s.{}.dat'.format(NT,I),err_outs)
#np.savetxt('ANNDper_w1_HW13_params2_{}s.{}.dat'.format(NT,I),w_1)
#np.savetxt('ANNDper_w2_HW13_params2_{}s.{}.dat'.format(NT,I),w_2)
#np.savetxt('ANNDper_b_HW13_params2_{}s.{}.dat'.format(NT,I),b)
#np.savetxt('ANNDper_Wout_HW13_params2_{}s.{}.dat'.format(NT,I),Wouts,fmt=['%15f'])

print('Program finished at {}.'.format(datetime.now().time()))

