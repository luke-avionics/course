import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as mat
x=range(0,1000)
x=[i*1.0/999 for i in x]
e=np.random.normal(0,0.5,1000)
y=[]
for i in range(0,len(x)):
    y.append(x[i]+e[i])
mat.figure()
mat.scatter(x,y)
mat.title('Scatter plot of y')
mat.xlabel('x')
mat.ylabel('y')
mat.hold(True)



#same mehod as computing least square problem
a=(np.matmul(np.transpose(x),y)+np.matmul(np.transpose(y),x))*1.0/(2*np.matmul(np.transpose(x),x))
f=[a*i for i in x]
mat.plot(x,f,linewidth=3.0,color='r')

y2=[]
for i in range(0,len(x)):
    y2.append(30*(x[i]-0.25)**2*(x[i]-0.75)**2+e[i])
mat.figure()
mat.scatter(x,y2)
mat.hold(True)

X=np.empty([len(x),5],float)
for i in range(0,len(x)):
    for j in range(0,5):
        X[i,j]=x[i]**(j)
y2=np.reshape(y2,[len(y2),1])
a_ls=np.matmul(np.matmul(inv(np.matmul(np.transpose(X),X)),np.transpose(X)),y2)
f2=[a_ls[0]+a_ls[1]*i+a_ls[2]*i**2+a_ls[3]*i**3+a_ls[4]*i**4 for i in x]
mat.plot(x,f2,linewidth=3.0,color='r')
mat.show()

