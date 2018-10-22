m=1880.0
n=89121.0
p=1/(n+m)
q=1/(m+n)
d1=n*(1-q-p+p*q)-(n**2-n)*(p-p*q)-m*n*(q-p*q)
d2=m*(1-q-p+p*q)-(m**2-m)*(q-p*q)-m*n*(p-p*q)
print(d1)
print(d2)
