import csv
from numpy.linalg import inv,norm
import numpy as np
import matplotlib.pyplot as mat
import scipy.io as sio
b=[4.5,6]
b=np.reshape(b,(len(b),1))
x=np.zeros((len(b),1))
x_min=b

#||x-b||_2
#contant step size
max_iter=100
error_list=[] 
iter_r=0
for i in range(0,max_iter):
    ak=1
    x=x-ak*2*(x-b)/norm(x-b)
    #x=x-ak*np.multiply(np.sqrt(np.absolute(2*(x-b))),np.divide(2*(x-b),np.absolute(2*(x-b))))
    error_temp=norm(x_min-x)/norm(x_min)
    error_list.append(error_temp)
    if error_temp <=0.01:
        iter_r=i
        break
    iter_r=i

if iter_r==max_iter-1:
    print ('failed to converge')
mat.figure()
mat.plot(range(0,iter_r+1),error_list) 



x=np.zeros((len(b),1))


#(5/6)^k
max_iter=100
error_list=[]
iter_r=0
for i in range(0,max_iter):
    ak=(5/6)**i
    x=x-ak*2*(x-b)/norm(x-b)
    #x=x-ak*np.multiply(np.sqrt(np.absolute(2*(x-b))),np.divide(2*(x-b),np.absolute(2*(x-b))))
    error_temp=norm(x_min-x)/norm(x_min)
    error_list.append(error_temp)
    if error_temp <=0.01:
        iter_r=i
        break
    iter_r=i
if iter_r==max_iter-1:
    print ('failed to converge')

mat.figure()
#mat.hold(True)
mat.plot(range(0,iter_r+1),error_list)

x=np.zeros((len(b),1))


#1/(k+1)
max_iter=100
error_list=[]
iter_r=0
for i in range(0,max_iter):
    ak=1/(i+1)
    x=x-ak*2*(x-b)/norm(x-b)
    #x=x-ak*np.multiply(np.sqrt(np.absolute(2*(x-b))),np.divide(2*(x-b),np.absolute(2*(x-b))))
    error_temp=norm(x_min-x)/norm(x_min)
    error_list.append(error_temp)
    if error_temp <=0.01:
        iter_r=i
        break
    iter_r=i

if iter_r==max_iter-1:
    print ('failed to converge')
mat.figure()
mat.plot(range(0,iter_r+1),error_list)


x=np.zeros((len(b),1))

#||x-b||_2^2
#contant step size
max_iter=100
error_list=[]
iter_r=0
for i in range(0,max_iter):
    ak=1
    x=x-ak*2*(x-b)
    #x=x-ak*np.multiply(np.sqrt(np.absolute(2*(x-b))),np.divide(2*(x-b),np.absolute(2*(x-b))))
    error_temp=norm(x_min-x)/norm(x_min)
    error_list.append(error_temp)
  
    if error_temp <=0.01:
        iter_r=i
        break
    iter_r=i

if iter_r==max_iter-1:
    print ('failed to converge')
mat.figure()
mat.plot(range(0,iter_r+1),error_list)


mat.show()
    
