import numpy as np
print("\x1b[2J\x1b[1;1H") 

# Lendo o arquivo de saídas esperadas (target)
t=np.loadtxt('data/targets10.csv',delimiter=';',skiprows=0)

#t=np.loadtxt('targest16.csv',delimiter=';',skiprows=0)
vanterior=np.loadtxt('exp_neurons_100_lr_0.010000_act_tanh/vnovo.csv', delimiter=';', skiprows=0)
v0anterior=np.loadtxt('exp_neurons_100_lr_0.010000_act_tanh/v0novo.csv', delimiter=';', skiprows=0)
wanterior=np.loadtxt('exp_neurons_100_lr_0.010000_act_tanh/wnovo.csv', delimiter=';', skiprows=0)
w0anterior=np.loadtxt('exp_neurons_100_lr_0.010000_act_tanh/w0novo.csv', delimiter=';', skiprows=0)

# print(vanterior)
print(v0anterior)
# print(wanterior)
# print(w0anterior)


(vent,neur)=np.shape(vanterior)
(vsai,numclasses)=np.shape(t)
print(vent)
print(neur)
print(vsai)
print(numclasses)
limiar=0
zin=np.zeros((1,neur))
target=np.zeros((vsai,1))


###################### Limiarização


#### Teste da rede
aminicial=81
amtestedigitos=10
yteste=np.zeros((vsai,1))
k2='_'
k4='.txt'
cont=0
contcerto=0
#ordem=np.zeros(amostras)
for m in range(10):   
    k1=str(m)   
    for n in range(amtestedigitos):      
        k3a=n+aminicial
        k3=str(k3a)
        nome=k1+k2+k3+k4
        xteste=np.loadtxt('data/'+nome)
        for m2 in range(vsai):
            for n2 in range(neur):
                zin[0][n2]=np.dot(xteste,vanterior[:,n2])+v0anterior[n2]
            z=np.tanh(zin)
            yin=np.dot(z,wanterior)+w0anterior
            y=np.tanh(yin)
        for j in range(vsai):
            if yin[0][j]>=limiar:
                y[0][j]=1.0
            else:
                y[0][j]=-1.0
        for j in range(vsai):
            yteste[j][0]=y[0][j]
        
        for j in range(vsai):
            target[j][0]=t[j][m]
        soma=np.sum(y-target)
        if soma==0:
            contcerto=contcerto+1
        cont=cont+1
taxa=contcerto/cont
print(taxa)        


################## Distância Euclidiana
### Teste da rede
#aminicial=101
#amtestedigitos=35
#yteste=np.zeros((vsai,1))
#k2='_'
#k4='.txt'
#cont=0
#contcerto=0
#
#for m in range(10):   
#    k1=str(m)   
#    for n in range(amtestedigitos):      
#        k3a=n+aminicial
#        k3=str(k3a)
#        nome=k1+k2+k3+k4
#        xteste=np.loadtxt(nome)
#        for m2 in range(vsai):
#            for n2 in range(neur):
#                zin[0][n2]=np.dot(xteste,vanterior[:,n2])+v0anterior[n2]
#            z=np.tanh(zin)
#            yin=np.dot(z,wanterior)+w0anterior
#            y=np.tanh(yin)
#        disteuclidiana=np.zeros((1,numclasses))
#        for j in range(numclasses):
#            distaux=0
#            for m3 in range(vsai):
#                distaux=distaux+(y[0][m3]-t[m3][j])**2
#            disteuclidiana[0][j]=np.sqrt(distaux)
#        indice=disteuclidiana.argmin()
#        if indice==m:
#            contcerto=contcerto+1
#        cont=cont+1
#        
#taxa=contcerto/cont
#print(taxa)        
