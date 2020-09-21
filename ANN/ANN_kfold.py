# Program to test combinations of hyperparameters through k-fold validation (
# Architecture with least MAE after summed over each folding is the selection.


#Declerations
import numpy as np
import itertools

#Definitions
def sig(x):
        f=1/(1+np.exp(-x))
        return f

def sig_deriv(x): # Must feed in sig(x) for which you want the derivative.
	f=x*(1-x)
	return f
	
def ReLU(x):
        x[x<0]=0
        f=x
        return f

def ReLU_deriv(x):
        x[x<0]=0
        x[x>=0]=1
        f=x
        return f

def tanh(x):
        f=2*sig(2*x)-1
        return f

def tanh_deriv(x): # Must feed in tanh(x) for which you want the derivative.
	f=1-x**2
	return f



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

#User interface prompt
print('----------------------------------------------------------------------------------------------------\nWelcome to ANN K-fold validation algorithm for neural network architecture selection!\nRespond in the following manner;\n(1st option=0,2nd option=1,3rd etc): User enter---> 1st choice comma 2nd choice comma 3rd etc...\n')
#choice={'Layer':0,'Neuron':0,'Function':0,'Params':0,'Sample':0,'Regularization':0,'Feed':0,'Iterations':0,'StepSize':0,'Loss':0}

#choice['Layer']=input('Which number of hidden layers to include? (1=0,2=1): ')
#choice['Neuron']=input('How many neurons per layer to include? (5=0,15=1,50=2): ')
#choice['Function']=input('Activation functions to include? (Sigmoid=0,ReLU=1):  ')
#choice['Params']=input('Selection of only the param values vs also the covariance elements of params to include (params only=0, covariance=1): ')
#choice['Sample']=input('Distribution of training set based on output (Catalog Dist=0, Uniform Dist=1): ')
#choice['Regularization']=input('Regularization of inputs to include? (SIG(Mean0Std1)=0,TANH(Mean0Std1)=1): ')
#choice['Feed']=input('Feed-in of inputs to include? (Full=0,Batch=1,Stocastic=2): ')
#choice['Iterations']=input('Number of iterations to include? (10=0,100=1,1000=2): ')
#choice['StepSize']=input('Step size to include? (0.0001=0,0.001=1,0.01=2): ')
#choice['Loss']=input('Loss functions to include? (Chi2=0,Chi2Prob=1): ')

#s=[]
#for x,y in choice.iteritems():
#    if len(np.atleast_1d(y)) > 1:
#        s.append(x)
#M=len(s)
#mix=[]
#for i in range(0,M):
#	if i < M:
#		mix.append(choice[s[i]])
	
#perms=np.array([list(itertools.product(eval(str(mix)[1:-1])))])

Layer=input('Which number of hidden layers to include? (1=0,2=1): ')
Neuron=input('How many neurons per layer to include? (5=0,15=1,50=2): ')
Function=input('Activation functions to include? (Sigmoid=0,Tanh=1):  ')
Params=input('Selection of only the param values vs also the covariance elements of params to include (params only=0, covariance=1): ')
Sample=input('Distribution of training set based on output (Catalog Dist=0, Uniform Dist=1): ')
Regularization=input('Regularization of inputs to include? (SIG(Mean0Std1)=0,TANH(Mean0Std1)=1): ')
Feed=input('Feed-in of inputs to include? (Full=0,Batch=1,Stocastic=2): ')
Iterations=input('Number of iterations to include? (10=0,100=1,1000=2): ')
StepSize=input('Step size to include? (0.0001=0,0.001=1,0.01=2): ')
Loss=input('Loss functions to include? (Chi2=0,Chi2Prob=1): ')
print('------------------------------------------------------')

Perms=np.array([list(itertools.product(Layer,Neuron,Function,Params,Sample,Regularization,Feed,Iterations,StepSize,Loss))])
Perms=Perms[0,:,:]
Nperm=len(Perms)
np.savetxt('HyperPerms_2.dat',Perms,fmt='%5d')
print('{} combinations of hyperparameters.\n'.format(Nperm))


#^^^^^^^^^^^^ Need to reduce options to most interesting choices ^^^^^^^^^^^^^^^^^^#

from datetime import datetime
print('Program started at {}.\n--------------------------------------------'.format(datetime.now().time()))

#Create a loop over the entire program here that will make use of each row in Perms to set the architecture.
print('Loading and initializing data.....\n')
# Load Data
data=np.loadtxt('bolshoi_9params.dat',unpack=False)
np.random.shuffle(data)
N=len(data)

#For Log param only inputs
data[:,0:9]=np.log(data[:,0:9])

# For Log param covariance inputs
avg=np.average(data[:,0:9],axis=0)
cov=np.zeros((N,9,9))
INcov=np.zeros((N,45))
OUTcov=np.zeros(N)
for i in range(0,N):
        	if i < N:
                	for j in range(0,9):
                        	if j < 9:
                                	for k in range(0,9):
                                        	if k < 9:
                                                	cov[i,j,k]=(data[i,j]-avg[j])*(data[i,k]-avg[k])
			INcov[i,:]=np.triu(cov[i,:,:])[(np.triu(cov[i,:,:])!=0)].flatten() 
			OUTcov[i]=data[i,-1]
dataCOV=np.vstack((INcov.T,OUTcov)).T

print('Begin testing of hyperparameter combinations at {}.\n------------------------------------------------'.format(datetime.now().time()))
MAE=np.zeros(Nperm)
for i in range(0,Nperm):
	if i < Nperm:
		N=len(data)
		print('Begin combination {} at {}----'.format(Perms[i,:],datetime.now().time()))
		#Split into k-fold sets (3 to train on and 1 for validation). Either for just params or covariance elements for catalog dist. or uniform dist.
		if Perms[i,3]==0:
			if Perms[i,4]==0:		
                       		NI=9
				sel=N/5
				DATA1=data[0:sel,:]
				DATA2=data[sel:2*sel,:]
				DATA3=data[2*sel:3*sel,:]
				DATA4=data[3*sel:4*sel,:]
				EVAL=data[4*sel:5*sel,:]
			else:
				NI=9
			        bins=np.linspace(data[:,-1].min(),data[:,-1].max(),23)
			        hist,edge=np.histogram(data[:,-1],bins)
			        digit=np.digitize(data[:,-1],bins)
			        binselect=np.floor(hist/5.).astype(int)
			        minim=int(binselect.min())
			        L=len(binselect)
			        DATA1=np.zeros((L*minim,NI+1))
        			DATA2=np.zeros((L*minim,NI+1))
        			DATA3=np.zeros((L*minim,NI+1))
        			DATA4=np.zeros((L*minim,NI+1))
				EVAL=np.zeros((L*minim,NI+1))
        			for t in range(0,(L*minim)):
                			if t < (L*minim):
                        			DATA1[t,:]=data[digit==(t+1)][0,:]
                       		 		DATA2[t,:]=data[digit==(t+1)][1,:]
                        			DATA3[t,:]=data[digit==(t+1)][2,:]
                        			DATA4[t,:]=data[digit==(t+1)][3,:]
						EVAL[t,:]=data[digit==(t+1)][4,:]
        			np.random.shuffle(DATA1)
        			np.random.shuffle(DATA2)
        			np.random.shuffle(DATA3)
        			np.random.shuffle(DATA4)
				np.random.shuffle(EVAL)
               	else:
			if Perms[i,4]==0:
                        	NI=45
				sel=N/45
                        	DATA1=dataCOV[0:sel,:]
                        	DATA2=dataCOV[sel:2*sel,:]
                        	DATA3=dataCOV[2*sel:3*sel,:]
                        	DATA4=dataCOV[3*sel:4*sel,:]
				EVAL=dataCOV[4*sel:5*sel,:]
			else:
				NI=45
                                bins=np.linspace(dataCOV[:,-1].min(),dataCOV[:,-1].max(),23)
                                hist,edge=np.histogram(dataCOV[:,-1],bins)
                                digit=np.digitize(dataCOV[:,-1],bins)
                                binselect=np.floor(hist/5.).astype(int)
                                minim=int(binselect.min())
                                L=len(binselect)
                                DATA1=np.zeros((L*minim,NI+1))
                                DATA2=np.zeros((L*minim,NI+1))
                                DATA3=np.zeros((L*minim,NI+1))
                                DATA4=np.zeros((L*minim,NI+1))
				EVAL=np.zeros((L*minim,NI+1))
                                for t in range(0,(L*minim)):
                                        if t < (L*minim):
                                                DATA1[t,:]=dataCOV[digit==(t+1)][0,:]
                                                DATA2[t,:]=dataCOV[digit==(t+1)][1,:]
                                                DATA3[t,:]=dataCOV[digit==(t+1)][2,:]
                                                DATA4[t,:]=dataCOV[digit==(t+1)][3,:]
						EVAL[t,:]=dataCOV[digit==(t+1)][4,:]
                                np.random.shuffle(DATA1)
                                np.random.shuffle(DATA2)
                                np.random.shuffle(DATA3)
                                np.random.shuffle(DATA4)
				np.random.shuffle(EVAL)
		Neval=len(EVAL)
		MAE_val=np.zeros(4)
		# Run hyperparameter combination for 4 folds.
		for j in range(0,4): 
			if j < 4:
				if j==0:
					DATA=np.hstack((DATA1.T,DATA2.T,DATA3.T)).T
					VAL=DATA4
                                if j==1:
                                        DATA=np.hstack((DATA2.T,DATA3.T,DATA4.T)).T
                                        VAL=DATA1
                                if j==2:
                                        DATA=np.hstack((DATA1.T,DATA3.T,DATA4.T)).T
                                        VAL=DATA2
                                if j==3:
                                        DATA=np.hstack((DATA1.T,DATA2.T,DATA4.T)).T
                                        VAL=DATA3
				N=len(DATA)
				Nval=len(VAL)
				# Inverse weight on loss function according to frequency of training set in data. (no or yes)
				if Perms[i,9]==0:
					pout=np.zeros(N)+1
				else:
					bins=np.linspace(DATA[:,-1].min(),DATA[:,-1].max(),40)
					hist,edge=np.histogram(DATA[:,-1],bins)
					p=1/hist.astype(float)
					digitout=np.digitize(DATA[:,-1],bins[0:-1])-1
					new_maxp=1000.
					new_minp=1.
					p=((p-min(p))/(max(p)-min(p)))*(new_maxp-new_minp)+new_minp
					pout=p[digitout]

				# Regularize the inputs 
				nDATA=np.zeros((N,(NI+1)))
				nVAL=np.zeros((Nval,NI+1))
				nEVAL=np.zeros((Neval,NI+1))
				if Perms[i,5]==0:	
					for ni in range(0,(NI)):
        					if ni < (NI) :
                					nDATA[:,ni]=sig(((DATA[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni])))
							nVAL[:,ni]=sig(((VAL[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni])))
                                                        nEVAL[:,ni]=sig(((EVAL[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni])))
				else:
                        		for ni in range(0,(NI)):
                               			if ni < (NI) :
                                       			nDATA[:,ni]=tanh(((DATA[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni])))
                                                        nVAL[:,ni]=tanh(((VAL[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni])))
                                                        nEVAL[:,ni]=tanh(((EVAL[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni])))
				nDATA[:,-1]=DATA[:,-1]
				nVAL[:,-1]=VAL[:,-1]
				nEVAL[:,-1]=EVAL[:,-1]
				if Perms[i,0]==0:
				        if Perms[i,1]==0:
						K=5
					if Perms[i,1]==1:
						K=15
					if Perms[i,1]==2:
						K=50
					b=np.random.normal(0,1,K)
				        r1=1/np.sqrt(NI)
        				w_1=np.random.normal(0,r1,[NI,K])
        				r2=1/np.sqrt(K)
        				w_2=np.random.normal(0,r2,K)
					if Perms[i,6]==0:
						NT=N
						NTepoch=1
					if Perms[i,6]==1:
						if Perms[i,4]==1:
							NT=10
							NTepoch=np.floor(N/NT).astype(int)
						else:
							NT=1000
							NTepoch=np.floor(N/NT).astype(int)
					if Perms[i,6]==2:
						NT=1
						NTepoch=N
					if Perms[i,7]==0:
						IT=10
					if Perms[i,7]==1:
						IT=100
					if Perms[i,7]==2:
						IT=1000
					if Perms[i,8]==0:
						eta=0.0001
                                        if Perms[i,8]==1:
                                                eta=0.001
                                        if Perms[i,8]==2:
                                                eta=0.01


				        for nti in range(0,NTepoch):
                				if nti < NTepoch:
                        				Nin=nDATA[nti*NT:(nti+1)*NT,0:NI]
                        				Nout=nDATA[nti*NT:(nti+1)*NT,NI]
							Pout=pout[nti*NT:(nti+1)*NT]
                        				for it in range(0,IT):
                                				if it < IT:
									x=np.dot(Nin,w_1)+b
									if Perms[i,2]==0:
                                        					t1=sig(x)
                                        				else:
										t1=tanh(x)
									q=np.sum(w_2*t1,axis=1)

				                                        if Perms[i,2]==0:
										del_w_2=np.sum(-eta*Pout*(q-Nout)*t1.T,axis=1)
                        				        	        del_w_1=np.dot((-eta*Pout*(q-Nout)*(sig_deriv(t1)*w_2).T),Nin).T
                                					        del_b=np.sum(-eta*Pout*(q-Nout)*(sig_deriv(t1)*w_2).T,axis=1) 
										w_1=w_1+del_w_1
                                        					w_2=w_2+del_w_2
                                       						b=b+del_b
									else:
                                                                                del_w_2=np.sum(-eta*Pout*(q-Nout)*t1.T,axis=1)
                                                                                del_w_1=np.dot((-eta*Pout*(q-Nout)*(tanh_deriv(t1)*w_2).T),Nin).T
                                                                                del_b=np.sum(-eta*Pout*(q-Nout)*(tanh_deriv(t1)*w_2).T,axis=1)
                                                                                w_1=w_1+del_w_1
                                                                                w_2=w_2+del_w_2
                                                                                b=b+del_b
					VALin=nVAL[:,0:NI]
					VALout=nVAL[:,NI]
                                        xVAL=np.dot(VALin,w_1)+b
                                        if Perms[i,2]==0:
                                        	t1=sig(xVAL)
                                        else:
                                                t1=tanh(xVAL)
                                        qVAL=np.sum(w_2*t1,axis=1)
					MAE_val[j]=np.sum(abs(qVAL-VALout))/Nval			

                                if Perms[i,0]==1:
                                        if Perms[i,1]==0:
                                                K=5
                                        if Perms[i,1]==1:
                                                K=15
                                        if Perms[i,1]==2:
                                                K=50
				        b1=np.random.normal(0,1,K)
        				b2=np.random.normal(0,1,K)
        				r1=1/np.sqrt(NI)
                        		w_1=np.random.normal(0,r1,[NI,K])
					r2=1/np.sqrt(K)
                        		w_2=np.random.normal(0,r2,[K,K])
        				w_3=np.random.normal(0,r2,K)
                                        if Perms[i,6]==0:
                                                NT=N
                                                NTepoch=1
                                        if Perms[i,6]==1:
                                                if Perms[i,4]==1:
                                                        NT=10
                                                        NTepoch=np.floor(N/NT).astype(int)
                                                else:
                                                        NT=1000
                                                        NTepoch=np.floor(N/NT).astype(int)
                                        if Perms[i,6]==2:
                                                NT=1
                                                NTepoch=N
                                        if Perms[i,7]==0:
                                                IT=10
                                        if Perms[i,7]==1:
                                                IT=100
                                        if Perms[i,7]==2:
                                                IT=1000
                                        if Perms[i,8]==0:
                                                eta=0.0001
                                        if Perms[i,8]==1:
                                                eta=0.001
                                        if Perms[i,8]==2:
                                                eta=0.01
			
				        for nti in range(0,NTepoch):
                				if nti < NTepoch:
                        				Nin=nDATA[nti*NT:(nti+1)*NT,0:NI]
                        				Nout=nDATA[nti*NT:(nti+1)*NT,NI]
                        				Pout=pout[nti*NT:(nti+1)*NT]
                        				for it in range(0,IT):
                                				if it < IT:
									x1=np.dot(Nin,w_1)+b1
                                                                        if Perms[i,2]==0:
                                                                                t1=sig(x1)
										x2=np.dot(t1,w_2)+b2
										t2=sig(x2)
                                                                        else:
                                                                                t1=tanh(x1)
										x2=np.dot(t1,w_2)+b2
										t2=tanh(x2)
                                                                        q=np.sum(w_3*t2,axis=1)

                                                                        if Perms[i,2]==0:
					                                        del_w_1=np.dot(((np.dot(((sig_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))).T,w_2.T))*(sig_deriv(t1))).T,Nin).T
                                        					del_w_3=np.sum(-eta*Pout*(q-Nout)*t2.T,axis=1)
                                        					del_w_2=np.dot(((sig_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))),t1).T
                                        					del_b1=np.sum((np.dot(((sig_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))).T,w_2.T))*(sig_deriv(t1)),axis=0)
                                        					del_b2=np.sum(-eta*Pout*(q-Nout)*(sig_deriv(t2)*w_3).T,axis=1)
										w_1=w_1+del_w_1
                                        					w_2=w_2+del_w_2
                                        					w_3=w_3+del_w_3
                                        					b1=b1+del_b1
                                        					b2=b2+del_b2
									else:
                                                                                del_w_1=np.dot(((np.dot(((tanh_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))).T,w_2.T))*(tanh_deriv(t1))).T,Nin).T
                                                                                del_w_3=np.sum(-eta*Pout*(q-Nout)*t2.T,axis=1)
                                                                                del_w_2=np.dot(((tanh_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))),t1).T
                                                                                del_b1=np.sum((np.dot(((tanh_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))).T,w_2.T))*(tanh_deriv(t1)),axis=0)
                                                                                del_b2=np.sum(-eta*Pout*(q-Nout)*(tanh_deriv(t2)*w_3).T,axis=1)
                                                                                w_1=w_1+del_w_1
                                                                                w_2=w_2+del_w_2
                                                                                w_3=w_3+del_w_3
                                                                                b1=b1+del_b1
                                                                                b2=b2+del_b2

                                        VALin=nVAL[:,0:NI]
                                        VALout=nVAL[:,NI]
                                        xVAL1=np.dot(VALin,w_1)+b1
                                        if Perms[i,2]==0:
                               	        	t1=sig(xVAL1)
                                                xVAL2=np.dot(t1,w_2)+b2
                                                t2=sig(xVAL2)
                                        else:
                                                t1=tanh(xVAL1)
                                                xVAL2=np.dot(t1,w_2)+b2
                                                t2=tanh(xVAL2)
                                        qVAL=np.sum(w_3*t2,axis=1)
                                        MAE_val[j]=np.sum(abs(qVAL-VALout))/Nval
                MAE[i]=np.sum(MAE_val)/4.
		print('MAE= {}\n'.format(MAE[i]))

np.savetxt('HyperMAE_2.dat',MAE)
print('Finished running through hyperparameter combinations at {}.\n'.format(datetime.now().time()))

print('-------------Evaluating---------------\n')
Perm_sel=Perms[MAE==MAE.min()]
if Perm_sel[3]==0:
	if Perm_sel[4]==0:
        	NI=9
                sel=N/5
                DATA1=data[0:sel,:]
                DATA2=data[sel:2*sel,:]
                DATA3=data[2*sel:3*sel,:]
                DATA4=data[3*sel:4*sel,:]
                EVAL=data[4*sel:5*sel,:]
        else:
                NI=9
                bins=np.linspace(data[:,-1].min(),data[:,-1].max(),23)
                hist,edge=np.histogram(data[:,-1],bins)
                digit=np.digitize(data[:,-1],bins)
                binselect=np.floor(hist/5.).astype(int)
                minim=int(binselect.min())
                L=len(binselect)
                DATA1=np.zeros((L*minim,NI+1))
                DATA2=np.zeros((L*minim,NI+1))
                DATA3=np.zeros((L*minim,NI+1))
                DATA4=np.zeros((L*minim,NI+1))
                EVAL=np.zeros((L*minim,NI+1))
                for t in range(0,(L*minim)):
 	               if t < (L*minim):
        	               DATA1[t,:]=data[digit==(t+1)][0,:]
                               DATA2[t,:]=data[digit==(t+1)][1,:]
                               DATA3[t,:]=data[digit==(t+1)][2,:]
                               DATA4[t,:]=data[digit==(t+1)][3,:]
                               EVAL[t,:]=data[digit==(t+1)][4,:]
                np.random.shuffle(DATA1)
                np.random.shuffle(DATA2)
                np.random.shuffle(DATA3)
                np.random.shuffle(DATA4)
                np.random.shuffle(EVAL)
else:
	if Perm_sel[4]==0:
        	NI=45
                sel=N/45
                DATA1=dataCOV[0:sel,:]
                DATA2=dataCOV[sel:2*sel,:]
                DATA3=dataCOV[2*sel:3*sel,:]
                DATA4=dataCOV[3*sel:4*sel,:]
                EVAL=dataCOV[4*sel:5*sel,:]
        else:
                NI=45
                bins=np.linspace(dataCOV[:,-1].min(),dataCOV[:,-1].max(),23)
                hist,edge=np.histogram(dataCOV[:,-1],bins)
                digit=np.digitize(dataCOV[:,-1],bins)
                binselect=np.floor(hist/5.).astype(int)
                minim=int(binselect.min())
                L=len(binselect)
                DATA1=np.zeros((L*minim,NI+1))
                DATA2=np.zeros((L*minim,NI+1))
                DATA3=np.zeros((L*minim,NI+1))
                DATA4=np.zeros((L*minim,NI+1))
                EVAL=np.zeros((L*minim,NI+1))
                for t in range(0,(L*minim)):
     	        	if t < (L*minim):
        	        	DATA1[t,:]=dataCOV[digit==(t+1)][0,:]
                           	DATA2[t,:]=dataCOV[digit==(t+1)][1,:]
                           	DATA3[t,:]=dataCOV[digit==(t+1)][2,:]
                           	DATA4[t,:]=dataCOV[digit==(t+1)][3,:]
                           	EVAL[t,:]=dataCOV[digit==(t+1)][4,:]
                np.random.shuffle(DATA1)
                np.random.shuffle(DATA2)
                np.random.shuffle(DATA3)
                np.random.shuffle(DATA4)
                np.random.shuffle(EVAL)
Neval=len(EVAL)
DATA=np.hstack((DATA1.T,DATA2.T,DATA3.T,DATA4.T)).T
N=len(DATA)
# Inverse weight on loss function according to frequency of training set in data. (no or yes)
if Perm_sel[9]==0:
	pout=np.zeros(N)+1
else:
	bins=np.linspace(DATA[:,-1].min(),DATA[:,-1].max(),40)
        hist,edge=np.histogram(DATA[:,-1],bins)
        p=1/hist.astype(float)
        digitout=np.digitize(DATA[:,-1],bins[0:-1])-1
        new_maxp=100.
        new_minp=1.
        p=((p-min(p))/(max(p)-min(p)))*(new_maxp-new_minp)+new_minp
        pout=p[digitout]
# Regularize the inputs 
nDATA=np.zeros((N,(NI+1)))
nEVAL=np.zeros((Neval,NI+1))
standardIN=np.zeros(((NI),2))
if Perm_sel[5]==0:
	for ni in range(0,(NI)):
        	if ni < (NI) :
                	nDATA[:,ni]=sig(((DATA[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni]))) 
                        nEVAL[:,ni]=sig(((EVAL[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni])))
			standardIN[ni,0]=np.mean(DATA[:,ni])
			standardIN[ni,1]=np.std(DATA[:,ni])
	np.savetxt('Standard_Hyper2.dat',standardIN)
else:
	for ni in range(0,(NI)):
        	if ni < (NI) :
                	nDATA[:,ni]=tanh(((DATA[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni]))) 
                        nEVAL[:,ni]=tanh(((EVAL[:,ni]-np.mean(DATA[:,ni]))/np.std(DATA[:,ni])))
nDATA[:,-1]=DATA[:,-1]
nEVAL[:,-1]=EVAL[:,-1]
if Perm_sel[0]==0:
	if Perm_sel[1]==0:
        	K=5
        if Perm_sel[1]==1:
                K=15
        if Perm_sel[1]==2:
                K=50
        b=np.random.normal(0,1,K)
        r1=1/np.sqrt(NI)
        w_1=np.random.normal(0,r1,[NI,K])
        r2=1/np.sqrt(K)
        w_2=np.random.normal(0,r2,K)
        if Perm_sel[6]==0:
        	NT=N
                NTepoch=1
        if Perm_sel[6]==1:
           	if Perm_sel[4]==1:
                	NT=10
                        NTepoch=np.floor(N/NT).astype(int)
                else:
                        NT=1000
                        NTepoch=np.floor(N/NT).astype(int)
        if Perm_sel[6]==2:
        	NT=1
                NTepoch=N
        if Perm_sel[7]==0:
                IT=10
        if Perm_sel[7]==1:
                IT=100
        if Perm_sel[7]==2:
                IT=1000
        if Perm_sel[8]==0:
                eta=0.0001
        if Perm_sel[8]==1:
                eta=0.001
        if Perm_sel[8]==2:
                eta=0.01

        for nti in range(0,NTepoch):
        	if nti < NTepoch:
                	Nin=nDATA[nti*NT:(nti+1)*NT,0:NI]
                        Nout=nDATA[nti*NT:(nti+1)*NT,NI]
                        Pout=pout[nti*NT:(nti+1)*NT]
                        for it in range(0,IT):
                        	if it < IT:
                                	x=np.dot(Nin,w_1)+b
                                        if Perm_sel[2]==0:
                                        	t1=sig(x)
                                        else:
                                                t1=tanh(x)
                                        q=np.sum(w_2*t1,axis=1)

                                        if Perm_sel[2]==0:
                                        	del_w_2=np.sum(-eta*Pout*(q-Nout)*t1.T,axis=1)
                                                del_w_1=np.dot((-eta*Pout*(q-Nout)*(sig_deriv(t1)*w_2).T),Nin).T
                                                del_b=np.sum(-eta*Pout*(q-Nout)*(sig_deriv(t1)*w_2).T,axis=1)
                                                w_1=w_1+del_w_1
                                                w_2=w_2+del_w_2
                                                b=b+del_b
                                        else:
                                                del_w_2=np.sum(-eta*Pout*(q-Nout)*t1.T,axis=1)
                                                del_w_1=np.dot((-eta*Pout*(q-Nout)*(tanh_deriv(t1)*w_2).T),Nin).T
                                                del_b=np.sum(-eta*Pout*(q-Nout)*(tanh_deriv(t1)*w_2).T,axis=1)
                                                w_1=w_1+del_w_1
                                                w_2=w_2+del_w_2
                                                b=b+del_b
                                       
        EVALin=nEVAL[:,0:NI]
	EVALout=nEVAL[:,NI]
        xEVAL=np.dot(EVALin,w_1)+b
        if Perm_sel[2]==0:
        	t1=sig(xEVAL)
        else:
              	t1=tanh(xEVAL)
        qEVAL=np.sum(w_2*t1,axis=1)
      	MAE_eval=np.sum(abs(qEVAL-EVALout))/Neval

if Perm_sel[0]==1:
	if Perm_sel[1]==0:
        	K=5
        if Perm_sel[1]==1:
                K=15
        if Perm_sel[1]==2:
                K=50
        b1=np.random.normal(0,1,K)
        b2=np.random.normal(0,1,K)
        r1=1/np.sqrt(NI)
        w_1=np.random.normal(0,r1,[K,NI])
        r2=1/np.sqrt(K)
        w_2=np.random.normal(0,r2,[K,K])
        w_3=np.random.normal(0,r2,K)
        if Perm_sel[6]==0:
        	NT=N
                NTepoch=1
        if Perm_sel[6]==1:
        	if Perm_sel[4]==1:
                	NT=10
                        NTepoch=np.floor(N/NT).astype(int)
                else:
                        NT=1000
                        NTepoch=np.floor(N/NT).astype(int)
        if Perm_sel[6]==2:
                 NT=1
                 NTepoch=N
        if Perm_sel[7]==0:
                 IT=10
        if Perm_sel[7]==1:
                 IT=100
        if Perm_sel[7]==2:
                 IT=1000
        if Perm_sel[8]==0:
                 eta=0.0001
        if Perm_sel[8]==1:
                 eta=0.001
        if Perm_sel[8]==2:
                 eta=0.01

        for nti in range(0,NTepoch):
        	if nti < NTepoch:
                	Nin=nDATA[nti*NT:(nti+1)*NT,0:NI]
                        Nout=nDATA[nti*NT:(nti+1)*NT,NI]
                        Pout=pout[nti*NT:(nti+1)*NT]
                        for it in range(0,IT):
                        	if it < IT:
                                	x1=np.dot(Nin,w_1)+b1
                                        if Perm_sel[2]==0:
                                        	t1=sig(x1)
                                                x2=np.dot(t1,w_2)+b2
                                                t2=sig(x2)
                                        else:
                                                t1=tanh(x1)
                                                x2=np.dot(t1,w_2)+b2
                                                t2=tanh(x2)
                                        q=np.sum(w_3*t2,axis=1)

                                        if Perm_sel[2]==0:
                                        	del_w_1=np.dot(((np.dot(((sig_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))).T,w_2.T))*(sig_deriv(t1))).T,Nin).T
                                                del_w_3=np.sum(-eta*Pout*(q-Nout)*t2.T,axis=1)
                                                del_w_2=np.dot(((sig_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))),t1).T
                                                del_b1=np.sum((np.dot(((sig_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))).T,w_2.T))*(sig_deriv(t1)),axis=0)
                                                del_b2=np.sum(-eta*Pout*(q-Nout)*(sig_deriv(t2)*w_3).T,axis=1)
                                                w_1=w_1+del_w_1
                                                w_2=w_2+del_w_2
                                                w_3=w_3+del_w_3
                                                b1=b1+del_b1
                                                b2=b2+del_b2
                                        else:
                                                del_w_1=np.dot(((np.dot(((tanh_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))).T,w_2.T))*(tanh_deriv(t1))).T,Nin).T
                                                del_w_3=np.sum(-eta*Pout*(q-Nout)*t2.T,axis=1)
                                                del_w_2=np.dot(((tanh_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))),t1).T
                                                del_b1=np.sum((np.dot(((tanh_deriv(t2)*w_3).T*(-eta*Pout*(q-Nout))).T,w_2.T))*(tanh_deriv(t1)),axis=0)
                                                del_b2=np.sum(-eta*Pout*(q-Nout)*(tanh_deriv(t2)*w_3).T,axis=1)
                                                w_1=w_1+del_w_1
                                                w_2=w_2+del_w_2
                                                w_3=w_3+del_w_3
                                                b1=b1+del_b1
                                                b2=b2+del_b2
	minmax=np.zeros((NI,2))
        for j in range(0,NI):
                if j < NI:
                        minmax[j,0]=nDATA[0:NTepoch*NT,j].min()
                        minmax[j,1]=nDATA[0:NTepoch*NT,j].max()

	EVALin=nEVAL[:,0:NI]
        EVALout=nEVAL[:,NI]
        xEVAL1=np.dot(EVALin,w_1)+b1
        if Perm_sel[2]==0:
        	t1=sig(xEVAL1)
                xEVAL2=np.dot(t1,w_2)+b2
                t2=sig(xEVAL2)
        else:
                t1=tanh(xEVAL1)
                xEVAL2=np.dot(t1,w_2)+b2
                t2=tanh(xEVAL2)
        qEVAL=np.sum(w_3*t2,axis=1)
        MAE_eval=np.sum(abs(qEVAL-EVALout))/Neval
EVAL_result=np.hstack((Perm_sel.T,MAE_eval))
np.savetxt('HyperEVAL_2.dat',EVAL_result)
print('Evaluation Complete!\nProgram Finished at {}.'.format(datetime.now().time()))




