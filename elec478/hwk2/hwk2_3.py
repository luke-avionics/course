import csv
from numpy.linalg import inv,norm
import numpy as np
import matplotlib.pyplot as mat
import scipy.io as sio

#cross validation
def cross_val(X,y,k,lambd):
    error=[]
    for i in range(0,k):
        if i!=k-1:
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
#calculate mean variance of the data
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
#prepare identity matrix to eliminate the penalization of biase term
I=np.identity(np.shape(X)[1])
theta_ind=np.empty((len(lambd),np.shape(X)[1]))
I[0][0]=0 
counter=0
#calculate theta for each lambda
for i in lambd:
    theta_ind[counter,:]=np.reshape(np.absolute(inv(np.transpose(X)@X+i*I)@np.transpose(X)@y),(1,np.shape(X)[1]))
    theta_r.append(norm(inv(np.transpose(X)@X+i*I)@np.transpose(X)@y))
    counter+=1
theta_l=norm(inv(np.transpose(X)@X)@np.transpose(X)@y)*np.ones((np.shape(lambd)[0],1))
#ploting ridge regression results
mat.figure()
mat.loglog(lambd,theta_r)
#plotting least square results
mat.hold(True)
mat.loglog(lambd,theta_l)
mat.title("Norm of theta vs. Lambda values")
mat.legend(['Ridge regression results','Least square results'])
mat.hold(False)

#plotting absolute values of each feature against lambda
#decide which features are important
lines = ["-","--","-.",":"]
mat.figure()
mat.hold(True)
for i in range(0,len(theta_ind[1,:])):
    indx_style=int(i/5)
    mat.semilogx(lambd,theta_ind[:,i],lines[indx_style])
mat.legend([str(i) for i in range(0,len(theta_ind[1,:]))])
mat.title("Abs of each feature vs. Lambda values")



#shuffle the data for cross validation
randomize = np.arange(len(X))
np.random.shuffle(randomize)
X = X[randomize]
y = y[randomize]


#do cross validation for eahc lambda values
error=[]
for i in lambd:
    error.append(cross_val(X,y,5,i))
#plot the cross validation results
mat.figure()
mat.semilogx(lambd,error)
mat.title('MSE vs. Lambda values (minimized when lambda='+str(round(lambd[np.argmin(error)],3))+')')
best_lambd=lambd[np.argmin(error)]
best_theta=inv(np.transpose(X)@X+best_lambd*I)@np.transpose(X)@y
print (best_theta)


mat.show()



