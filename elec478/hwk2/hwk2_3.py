import csv
from numpy.linalg import inv,norm
import numpy as np
import matplotlib.pyplot as mat
import scipy.io as sio

#cross validation
def cross_val(X,y,k,lambd):
    #randomize = np.arange(len(X))
    #np.random.shuffle(randomize)
    #X = X[randomize]
    #y = y[randomize]
    error=[]
    for i in range(0,k):
        if k!=4:
            X_test=X[i*int(len(X)/5):(i+1)*int(len(X)/5)]
            X_train=np.concatenate((X[:i*int(len(X)/5)],X[(i+1)*int(len(X)/5):]),axis=0)
            y_test=y[i*int(len(X)/5):(i+1)*int(len(X)/5)]
            y_train=np.concatenate((y[:i*int(len(X)/5)],y[(i+1)*int(len(X)/5):]),axis=0)
            #print(len(X_train))
            #print (len(y_train))
        else:
            X_test=X[i*int(len(X)/5):]
            X_train=X[:i*int(len(X)/5)]
            y_test=y[i*int(len(X)/5):]
            y_train=y[:i*int(len(X)/5)]
        I=np.identity(np.shape(X_train)[1])
        I[0][0]=0
        theta_r=inv(np.transpose(X_train)@X_train+lambd*I)@np.transpose(X_train)@y_train
        error_temp= norm(X_test@theta_r-y_test)/len(X_test)
        error.append(error_temp)
    return np.mean(error)



# load feature variables and their names
X = np.loadtxt("hitters.x.csv", delimiter=",", skiprows=1)
with open("hitters.x.csv", "r") as f:
    X_colnames = next(csv.reader(f))
# load salaries
y = np.loadtxt("hitters.y.csv", delimiter=",", skiprows=1)

X_mean=np.mean(X,axis=0)
X_std=np.std(X,axis=0)
y_mean=np.mean(y,axis=0)
#zero mean, unit variance
X=np.subtract(X,X_mean)
X=np.divide(X,X_std)
X=np.concatenate((np.ones((len(X),1)),X),axis=1)
#y=y-y_mean
y=np.reshape(y,(len(y),1))
lambd=np.logspace(-3,7,100,base=10)
theta_r=[]
I=np.identity(np.shape(X)[1])
I[0][0]=0 
for i in lambd:
    theta_r.append(norm(inv(np.transpose(X)@X+i*I)@np.transpose(X)@y))
theta_l=norm(inv(np.transpose(X)@X)@np.transpose(X)@y)*np.ones((np.shape(lambd)[0],1))
mat.figure()
mat.semilogx(lambd,theta_r)
mat.hold(True)
mat.semilogx(lambd,theta_l)
mat.hold(False)

randomize = np.arange(len(X))
np.random.shuffle(randomize)
X = X[randomize]
y = y[randomize]


error=[]
for i in lambd:
    error.append(cross_val(X,y,5,i))
mat.figure()
mat.semilogx(lambd,error)



mat.show()



