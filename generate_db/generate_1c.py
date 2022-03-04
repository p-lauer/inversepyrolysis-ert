#!/usr/bin/env python
# coding: utf-8

import time
import math
import numpy as np
from random import randrange, uniform
from multiprocessing import Process

def convert_tga_to_rk(rr,rt,hr,Y=1):
    #[rr]=/s,[rt]=°C,[hr]=K/min, [E]=J/mol, [A]=1/s
    rt=rt+273
    hr=hr/60
    E=(math.exp(1)*rr*8.3145*(rt**2))/hr
    A=(math.exp(1)*rr*math.exp(E/(8.3145*rt)))
    return(A,E)

def reactionrate(A,E,y,T):
    r=A*y*math.exp(-E/8.3145/T)
    return(r)

def calculate_reaction(A,E,Ys,T0=20,T1=550,HR=5.,dt=5):
    Y=Ys.copy()
    T0,T1=T0+273,T1+273
    x=np.zeros(((int((T1-T0)*(60./(HR*dt))+1),3)))
    x[:,0]=np.linspace(T0,T1,int((T1-T0)*(60./(HR*dt))+1))
    for i in x:
        k=0
        for c,t in enumerate(Y):
            if Y[c] == 0:
                pass
            else:
                r=reactionrate(A[c],E[c],Y[c],i[0])
                Y[c]=Y[c]-(r*dt)
                k=k+r
                if Y[c]<0:
                    r,Y[c]=0,0
        i[2]=sum(Y)
        i[1]=k
    x[:,0]=x[:,0]-273
    return(x.T)

def generate(dataset,nr):
    dataset=np.array(dataset)
    feat=[]
    np.savetxt("./labels{}k_5_10_30_40_1r_2ks_{}.csv".format(int(i/1000),int(nr)), np.copy(dataset.reshape(int((i/cores)*a),18)), delimiter=",")
    for ik,each1 in enumerate(np.copy(dataset)):
        each=each1.copy()
        temp1,masslossrate1,massloss1=calculate_reaction(each[3],each[4],each[2],HR=5.,dt=24,T0=Tstart, T1=Tend)
        each=each1.copy()
        temp2,masslossrate2,massloss2=calculate_reaction(each[3],each[4],each[2],HR=10.,dt=12,T0=Tstart, T1=Tend)
        each=each1.copy()
        temp3,masslossrate3,massloss3=calculate_reaction(each[3],each[4],each[2],HR=30., dt=4,T0=Tstart, T1=Tend)
        each=each1.copy()
        temp4,masslossrate4,massloss4=calculate_reaction(each[3],each[4],each[2],HR=40., dt=3,T0=Tstart, T1=Tend)
        feat.append((temp1,temp2,temp3,temp4,masslossrate1,masslossrate2,masslossrate3,masslossrate4,massloss1,massloss2,massloss3,massloss4))
    scale=len(temp1)
    feat=np.array(feat)
    print('{} done'.format(int(nr)))
    np.savetxt("./features{}k_5_10_30_40_1r_2ks_{}.csv".format(int(i/1000),nr), (np.copy(feat[:,4:8])).reshape(int(i/cores),scale*4), delimiter=",")

##################################
### Parameters defined by user ###
##################################

# i:        number of elements that will be generated
# cores:    number of cores    
# rrlimlow: lower boundary of peak reaction rate sampling (in /s)
# rrlimup:  upper boundary of peak reaction rate sampling (in /s)
# rtlimlow: lower boundary of peak reaction rate sampling (in °C)
# rtlimup:  upper boundary of peak reaction rate sampling (in °C)
# Tstart:   Start temperature of experiment (in °C)
# Tend:     End temperature of experiment (in °C)

# Note: i/cores must be an integer

i=6400000
cores=128
rrlimlow=0.001
rrlimup= 0.01
rtlimlow= 100
rtlimup= 500
Tstart=20
Tend=550

# RR/RT for 1 component
a=1
dataset=[]
for each in range(i):
    ia=1.
    ib=0
    ic=0
    RR,RT=round(uniform(rrlimlow,rrlimup),6),randrange(rtlimlow,rtlimup)
    RR2,RT2=0,0
    RR3,RT3=0,0
    E,A=[0,0,0],[0,0,0]
    A[0],E[0]=convert_tga_to_rk(RR,RT,5,ia)
    dataset.append(([RR,RR2,RR3],[RT,RT2,RT3],[ia,ib,ic],A,E,[ia,ib,ic]))


starttime=time.time()
dataset=np.array(dataset)
np.savetxt("./labels{}k_5_10_40_1r_2ks.csv".format(int(i/1000)), np.copy(dataset.reshape(i*a,18)), delimiter=",")

# calculate MLR

if __name__ == '__main__':
    processes = []
    for n in range(cores):
        p=Process(target=generate, args=(dataset[int((n*(i/cores))):int(((i/cores)+n*(i/cores)))],n,))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
endtime=time.time()
duration=endtime-starttime
print(duration)
