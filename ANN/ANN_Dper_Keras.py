import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras
from keras import regularizers
import matplotlib.pyplot as plt
import itertools
from keras import losses


##### Global Definitions #####
def sig(x):
        f=1/(1+np.exp(-x))
        return f

def tanh(x):
        f=2*sig(2*x)-1
        return f

TYPE='unimag_cent'
TYPE1='log'
TYPE2='unimag_cent'

data1=np.loadtxt('ILL_{}_data1.dat'.format(TYPE),unpack=False)
data2=np.loadtxt('ILL_{}_data2.dat'.format(TYPE),unpack=False)
data3=np.loadtxt('ILL_{}_data3.dat'.format(TYPE),unpack=False)
data4=np.loadtxt('ILL_{}_data4.dat'.format(TYPE),unpack=False)

data=np.vstack((data1,data2,data3,data4))
dateval=np.loadtxt('ILL_{}_eval.dat'.format(TYPE),unpack=False)
#data1=np.delete(data1,(0,6,7,8),axis=1)
#data2=np.delete(data2,(0,6,7,8),axis=1)
#data3=np.delete(data3,(0,6,7,8),axis=1)
#data4=np.delete(data4,(0,6,7,8),axis=1)

NI=10


param=np.loadtxt('ILL_{}_hyperparams.dat'.format(TYPE2),unpack=False)
error=np.loadtxt('ILL_{}_{}_errors.dat'.format(TYPE2,TYPE1),unpack=False)
params=param[error==error.min()][0].astype(int)


N=len(data)
Neval=len(dateval)
np.random.shuffle(data)
np.random.shuffle(dateval)
	
if params[0]==0:
	if TYPE=='unimass1' or TYPE=='unimass1_cent':
		DATA=np.zeros((N,NI+2))
		EVAL=np.zeros((Neval,NI+2))
	else:
		DATA=np.zeros((N,NI+1))
		EVAL=np.zeros((Neval,NI+1))
	standardDATAin=np.zeros((NI,2))
	for i in range(0,NI):
		if i < NI:
			standardDATAin[i,0]=np.mean(data[:,i])
			standardDATAin[i,1]=np.std(data[:,i])
			DATA[:,i]=sig((data[:,i]-standardDATAin[i,0])/standardDATAin[i,1])
			EVAL[:,i]=sig((dateval[:,i]-standardDATAin[i,0])/standardDATAin[i,1])
	if TYPE=='unimass1' or TYPE=='unimass1_cent':
		DATA[:,-2]=data[:,-2]
		DATA[:,-1]=data[:,-1]
		EVAL[:,-2]=dateval[:,-2]
		EVAL[:,-1]=dateval[:,-1]
	else:
		DATA[:,-1]=data[:,-1]
		EVAL[:,-1]=dateval[:,-1]

if params[0]==1:
	if TYPE=='unimass1' or TYPE=='unimass1_cent':
		DATA=np.zeros((N,NI+2))
		EVAL=np.zeros((Neval,NI+2))
	else:
		DATA=np.zeros((N,NI+1))
		EVAL=np.zeros((Neval,NI+1))
	standardDATAin=np.zeros((NI,2))
	for i in range(0,NI):
		if i < NI:
			standardDATAin[i,0]=np.mean(data[:,i])
			standardDATAin[i,1]=np.std(data[:,i])
			DATA[:,i]=tanh((data[:,i]-standardDATAin[i,0])/standardDATAin[i,1])
			EVAL[:,i]=tanh((dateval[:,i]-standardDATAin[i,0])/standardDATAin[i,1])		
	if TYPE=='unimass1' or TYPE=='unimass1_cent':
		DATA[:,-2]=data[:,-2]
		DATA[:,-1]=data[:,-1]
		EVAL[:,-2]=dateval[:,-2]
		EVAL[:,-1]=dateval[:,-1]
	else:
		DATA[:,-1]=data[:,-1]
		EVAL[:,-1]=dateval[:,-1]
					
x_train=DATA[:,0:NI]
y_train=DATA[:,NI:]
x_eval=EVAL[:,0:NI]
y_eval=EVAL[:,NI:]

					
if params[1]==0:
	neur=25
if params[1]==1:
	neur=50
if params[1]==2:
	neur=75
if params[2]==0:
	act='sigmoid'
if params[2]==1:
	act='tanh'

s=0

if params[3]==0:
	model = Sequential()
	model.add(Dense(units=neur, activation=act,use_bias=True,bias_regularizer=regularizers.l1(s),bias_initializer='RandomNormal', input_dim=NI))
if params[3]==1:
	model = Sequential()
	model.add(Dense(units=neur, activation=act,use_bias=True,bias_regularizer=regularizers.l1(s),bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,use_bias=True,bias_regularizer=regularizers.l1(s),bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,bias_regularizer=regularizers.l1(s),use_bias=True,bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,bias_regularizer=regularizers.l1(s),use_bias=True,bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,bias_regularizer=regularizers.l1(s),use_bias=True,bias_initializer='RandomNormal', input_dim=NI))
if params[3]==2:
	model = Sequential()
	model.add(Dense(units=neur, activation=act,use_bias=True,bias_regularizer=regularizers.l1(s),bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,use_bias=True,bias_regularizer=regularizers.l1(s),bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,use_bias=True,bias_regularizer=regularizers.l1(s),bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,use_bias=True,bias_regularizer=regularizers.l1(s),bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,use_bias=True,bias_regularizer=regularizers.l1(s),bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,bias_regularizer=regularizers.l1(s),use_bias=True,bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,bias_regularizer=regularizers.l1(s),use_bias=True,bias_initializer='RandomNormal', input_dim=NI))
	model.add(Dense(units=neur, activation=act,bias_regularizer=regularizers.l1(s),use_bias=True,bias_initializer='RandomNormal', input_dim=NI))

if TYPE=='unimass1' or TYPE=='unimass1_cent':
	model.add(Dense(units=2, activation='linear'))
else:
	model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')

if params[4]==0:
	epoch=50
	batch=int(N/epoch)
if params[4]==1:
	epoch=100
	batch=int(N/epoch)
if params[4]==2:
	epoch=200
	batch=int(N/epoch)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
hist=model.fit(x_train, y_train, epochs=epoch, validation_split=0.2,callbacks=[early_stopping],batch_size=batch)

error=model.evaluate(x_eval,y_eval,batch_size=Neval)

y_pred=model.predict(x_eval)

inEVAL=np.zeros((Neval,NI))
if NI>1:
	if params[0]==0:
		inEVAL[:,0:NI]=standardDATAin[:,0]-np.log(1/x_eval-1)*standardDATAin[:,1]
	if params[0]==1:
		inEVAL[:,0:NI]=-0.5*np.log(2/(x_eval+1)-1)*standardDATAin[:,1]+standardDATAin[:,0]	
else:
	if params[0]==0:
		inEVAL[:,0:NI]=standardDATAin[0][0]-np.log(1/x_eval-1)*standardDATAin[0][1]
	if params[0]==1:
		inEVAL[:,0:NI]=-0.5*np.log(2/(x_eval+1)-1)*standardDATAin[0][1]+standardDATAin[0][0]


plt.scatter(inEVAL[:,0],y_eval,s=0.1,color='k',label='Data')
plt.scatter(inEVAL[:,0],y_pred,s=0.1,color='r',label='ANN')
plt.legend(loc='upper left',fontsize=15)
plt.gca().invert_yaxis()
plt.show()


if TYPE=='unimass1' or TYPE=='unimass1_cent':
	plt.scatter(y_pred[:,0],y_eval[:,0],s=1,color='b')
	plt.plot(y_eval[:,0],y_eval[:,0],color='r')
	plt.title('Central Mag Prediction')
	plt.show()

	plt.scatter(y_pred[:,1],y_eval[:,1],s=1,color='b')
	plt.plot(y_eval[:,1],y_eval[:,1],color='r')
	plt.title('Central Color Prediction')
	plt.show()
if TYPE=='unimass2_cent':
	plt.scatter(y_pred,y_eval,s=1,color='b')
	plt.plot(y_eval,y_eval,color='r')
	plt.title('Central Mag Prediction')
	plt.show()
if TYPE=='unimass3_cent':
	plt.scatter(y_pred,y_eval,s=1,color='b')
	plt.plot(y_eval,y_eval,color='r')
	plt.title('Central Color Prediction')
	plt.show()
if TYPE=='unimag_cent':
	plt.scatter(y_pred,y_eval,s=1,color='b')
	plt.plot(y_eval,y_eval,color='r')
	plt.title('Central Mag Prediction')
	plt.show()
if TYPE=='unicolor_cent':
	plt.scatter(y_pred,y_eval,s=1,color='b')
	plt.plot(y_eval,y_eval,color='r')
	plt.title('Central Color Prediction')
	plt.show()




