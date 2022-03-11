#!/usr/bin/env python
# coding: utf-8

import time
import math
import numpy as np
import scipy
from scipy.signal import argrelextrema
from scipy.signal import peak_widths
from multiprocessing import Process
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
import multiprocessing
import pickle
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import scipy
import statistics
import csv
from random import randrange, uniform
from multiprocessing import Process
import multiprocessing
from scipy import interpolate
from scipy.spatial import ConvexHull
from matplotlib.image import NonUniformImage
from matplotlib.colors import LogNorm
from dot2tex import dot2tex
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from scipy.signal import argrelextrema
from scipy.signal import peak_widths
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from random import randrange, uniform
from scipy import interpolate
from scipy.spatial import ConvexHull
from multiprocessing import Process
import multiprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from matplotlib.patches import Rectangle
import itertools

def convert_tga_to_rk(rr,rt,hr,Y=1):
    #[rr]=/s,[rt]=°C,[hr]=K/min, [E]=J/mol, [A]=1/s
    rt=rt+273
    hr=hr/60
    E=(np.exp(1)*rr*8.3145*(rt**2))/hr
    A=(np.exp(1)*rr*np.exp((E/(8.3145*rt)),dtype=np.float128))
    return(A,E)

def reactionrate(A,E,y,T,n=1):
    r=A*y**n*math.exp(-E/8.3145/T)
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


def generate(datazet,nr):
    datazet=np.array(datazet)
    for ik,each in enumerate(datazet):
        temp1,masslossrate1,massloss1=calculate_reaction(np.exp(each[0:reactions]),each[3:6],each[6:9],HR=5.,dt=24)
        temp2,masslossrate2,massloss2=calculate_reaction(np.exp(each[0:reactions]),each[3:6],each[6:9],HR=10.,dt=12)
        temp3,masslossrate3,massloss3=calculate_reaction(np.exp(each[0:reactions]),each[3:6],each[6:9],HR=30., dt=4)
        temp4,masslossrate4,massloss4=calculate_reaction(np.exp(each[0:reactions]),each[3:6],each[6:9],HR=40., dt=3)
        return_list[len(datazet)*nr+ik]=temp1,temp2,temp3,temp4,masslossrate1,masslossrate2,masslossrate3,masslossrate4,massloss1,massloss2,massloss3,massloss4



def generate_2(datazet,labels,nr):
    datazet=np.array(datazet)
    for ik,each in enumerate(datazet):
        exampleset1=[]
        exampleset1.append(labels[ik])
        exampleset1=np.array(exampleset1)
        exampleset1[:,0:3]=np.exp(exampleset1[:,0:3])
        feat1=generate_3(exampleset1,3)
        exampleset3=[]
        for eacht in lista:
            exampleset3.append(each[eacht])
        exampleset3=np.array(exampleset3)
        exampleset3[:,0:3][exampleset3[:,0:3]>0]=np.exp(exampleset3[:,0:3][exampleset3[:,0:3]>0])
        feat3=generate_3(exampleset3,3)
        minim2=[]
        for eachs in feat3:
            minim2.append(np.sqrt(mean_squared_error(np.array(feat1[0,4:8]).reshape(1,-1),np.array(eachs[4:8]).reshape(1,-1)))/(np.max(np.array(feat1[0,4:8]).reshape(1,-1))-np.min(np.array(feat1[0,4:8]).reshape(1,-1))))
        return_list[len(datazet)*nr+ik]=np.min(minim2)
        exampleset3[:,0:3][exampleset3[:,0:3]>0]=np.log(exampleset3[:,0:3][exampleset3[:,0:3]>0])
        return_prediction[len(datazet)*nr+ik]=exampleset3[np.argsort(minim2)[0]]


def generate_3(dataset,nr):
    dataset=np.array(dataset)
    feat=[]
    for ik,each in enumerate(np.copy(dataset)):
        temp1,masslossrate1,massloss1=calculate_reaction(each[0:3],each[3:6],each[6:9],HR=5.,dt=24)
        temp2,masslossrate2,massloss2=calculate_reaction(each[0:3],each[3:6],each[6:9],HR=10.,dt=12)
        temp3,masslossrate3,massloss3=calculate_reaction(each[0:3],each[3:6],each[6:9],HR=30., dt=4)
        temp4,masslossrate4,massloss4=calculate_reaction(each[0:3],each[3:6],each[6:9],HR=40., dt=3)
        feat.append((temp1,temp2,temp3,temp4,masslossrate1,masslossrate2,masslossrate3,masslossrate4,massloss1,massloss2,massloss3,massloss4))
    scale=len(temp1)
    feat=np.array(feat)
    return feat


print('Loading Dataset....')
labels = []
features = []
x_array = list(range(20, 551, 2))
for i in [125, ]:
    feat = "../generate_db/features6400k_5_10_30_40_1r_2ks_" + str(i) + ".csv"
    lab = "../generate_db/labels6400k_5_10_30_40_1r_2ks_" + str(i) + ".csv"

    labels_temp = np.genfromtxt(lab, delimiter=',', dtype=np.float64)[:, 9:]
    features_temp = np.genfromtxt(feat, delimiter=',', dtype=np.float64)
    labels_temp[:, 0] = np.log(labels_temp[:, 0])
    features.append(features_temp)
    labels.append(labels_temp)

for i in [125, ]:
    feat = "../generate_db/features6400k_5_10_30_40_2r_2ks_" + str(i) + ".csv"
    lab = "../generate_db/labels6400k_5_10_30_40_2r_2ks_" + str(i) + ".csv"

    labels_temp = np.genfromtxt(lab, delimiter=',', dtype=np.float64)[:, 9:]
    features_temp = np.genfromtxt(feat, delimiter=',', dtype=np.float64)
    labels_temp[:, 0] = np.log(labels_temp[:, 0])
    labels_temp[:, 1] = np.log(labels_temp[:, 1])
    features.append(features_temp)
    labels.append(labels_temp)

for i in [125, ]:
    feat = "../generate_db/features6400k_5_10_30_40_3r_2ks_" + str(i) + ".csv"
    lab = "../generate_db/labels6400k_5_10_30_40_3r_2ks_" + str(i) + ".csv"

    labels_temp = np.genfromtxt(lab, delimiter=',', dtype=np.float64)[:, 9:]
    features_temp = np.genfromtxt(feat, delimiter=',', dtype=np.float64)
    labels_temp[:, 0] = np.log(labels_temp[:, 0])
    labels_temp[:, 1] = np.log(labels_temp[:, 1])
    labels_temp[:, 2] = np.log(labels_temp[:, 2])
    features.append(features_temp)
    labels.append(labels_temp)

features = np.array(features).reshape(samplesize, -1)
labels = np.array(labels).reshape(samplesize, -1)

predictions=np.genfromtxt('prediction.csv', delimiter=',', dtype=np.float64)

print('Dataset loaded....')

sort_prs=np.fliplr(np.argsort(labels[:,6:9]))
sort_prs=np.concatenate((sort_prs,sort_prs+3,sort_prs+6),axis=1)
sort_pre=np.fliplr(np.argsort(predictions[:,6:9]))
sort_pre=np.concatenate((sort_pre,sort_pre+3,sort_pre+6),axis=1)
complete_prediction=np.take_along_axis(predictions, sort_pre,axis=1)
complete_labels=np.take_along_axis(labels, sort_prs, axis=1)

####################
# Evaluation Table #
####################

n1=[]
n2=[]
n3=[]
alln=[]

n1.append(round(r2_score(complete_labels[0:50000,0],complete_prediction[0:50000,0]),2))
n2.append(r2_score(complete_labels[50000:100000,0:2],complete_prediction[50000:100000,0:2]))
n3.append(r2_score(complete_labels[100000:150000,0:3],complete_prediction[100000:150000,0:3]))

n1.append(round(r2_score(complete_labels[0:50000,3],complete_prediction[0:50000,3]),2))
n2.append(r2_score(complete_labels[50000:100000,3:5],complete_prediction[50000:100000,3:5]))
n3.append(r2_score(complete_labels[100000:150000,3:6],complete_prediction[100000:150000,3:6]))

n1.append('-')
n2.append(r2_score((complete_labels[50000:100000,6:8]),(complete_prediction[50000:100000,6:8])))
n3.append(r2_score((complete_labels[100000:150000,6:9]),(complete_prediction[100000:150000,6:9])))

n1.append(round(r2_score(complete_labels[0:50000,[0,3]],complete_prediction[0:50000,[0,3]]),2))
n2.append(r2_score(complete_labels[50000:100000,[0,1,3,4,6,7]],complete_prediction[50000:100000,[0,1,3,4,6,7]]))
n3.append(r2_score(complete_labels[100000:150000],complete_prediction[100000:150000]))

alln.append(r2_score(complete_labels[0:150000,0:3],complete_prediction[0:150000,0:3]))
alln.append(r2_score(complete_labels[0:150000,3:6],complete_prediction[0:150000,3:6]))
alln.append(r2_score(complete_labels[0:150000,6:9],complete_prediction[0:150000,6:9]))
alln.append(r2_score(complete_labels[0:150000],complete_prediction[0:150000]))

complete_labels[(complete_labels[0:150000]>0) & (complete_prediction[0:150000] > 0)].shape

data={'$n=1$':n1,
      '$n=2$':n2,
      '$n=3$':n3,
      'all $n$':alln}

df=pd.DataFrame(data,index=["$A_i$","$E_i$","$Y_i$","total"])
df.index.name='Parameter'
print((df.round(2)).to_latex(escape=False))
textfile = open("tab:evaluation.tex", "w")
a = textfile.write((df.round(2)).to_latex(escape=False))
textfile.close()

k=len(complete_prediction)
reactions=3
corez=100
        
starttime=time.time()
if __name__ == '__main__':
    processes = []
    manager=multiprocessing.Manager()
    return_list=manager.list([0]*k)
    for n in range(corez):
        p=Process(target=generate, args=(complete_prediction[int((n*(k/corez))):int(((k/corez)+n*(k/corez)))],n,))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
endtime=time.time()
duration=endtime-starttime
pred_feat=np.array(return_list)
prediction_features=(np.copy(pred_feat[:,4:8])).reshape(k,266*4)
dataset=[]

for it,each2 in enumerate(np.copy(features)):
    dataset.append(np.sqrt(mean_squared_error(features[it],prediction_features[it]))/(np.max(features[it])-np.min(features[it])))
complete_histogram=np.array(dataset)

r2_histogram=[]
k=50000
reactions=2
corez=200
r2_labels=complete_labels[50000:50000+k]
r2_prediction=complete_prediction[50000:50000+k]

a1=list(itertools.permutations([0,1]))
a2=list(itertools.permutations([3,4]))
a3=list(itertools.permutations([6,7]))

listb=np.array(list(itertools.product(a1,a2,a3))).reshape(8,-1)
lista=np.zeros((8,9)).astype(int)

lista[:,0:2]=listb[:,0:2]
lista[:,2]=2
lista[:,3:5]=listb[:,2:4]
lista[:,5]=5
lista[:,6:8]=listb[:,4:6]
lista[:,8]=8
print(lista)

starttime=time.time()
if __name__ == '__main__':
    processes = []
    manager=multiprocessing.Manager()
    return_list=manager.list([0]*k)
    return_prediction=manager.list([0]*k)
    for n in range(corez):
        p=Process(target=generate_2, args=(r2_prediction[int((n*(k/corez))):int(((k/corez)+n*(k/corez)))],r2_labels[int((n*(k/corez))):int(((k/corez)+n*(k/corez)))],n,))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
endtime=time.time()
duration=endtime-starttime
print(duration)

r2_histogram=np.array(return_list)
r2_results=np.array(return_prediction)

a1=list(itertools.permutations([0,1,2]))
a2=list(itertools.permutations([3,4,5]))
a3=list(itertools.permutations([6,7,8]))

lista=np.array(list(itertools.product(a1,a2,a3))).reshape(216,-1)
print(lista)

r3_histogram=[]
k=50000
reactions=3
corez=200

r3_labels=complete_labels[100000:100000+k]
r3_prediction=complete_prediction[100000:100000+k]

starttime=time.time()
if __name__ == '__main__':
    processes = []
    manager=multiprocessing.Manager()
    return_list=manager.list([0]*k)
    return_prediction=manager.list([0]*k)
    for n in range(corez):
        p=Process(target=generate_2, args=(r3_prediction[int((n*(k/corez))):int(((k/corez)+n*(k/corez)))],r3_labels[int((n*(k/corez))):int(((k/corez)+n*(k/corez)))],n,))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
endtime=time.time()
duration=endtime-starttime
print(duration)

r3_histogram=np.array(return_list)
r3_results=np.array(return_prediction)

complete_histogram2=np.concatenate((complete_histogram[0:50000],r2_histogram,r3_histogram))

#print(cfh)
scaling = 0.88
fontsize = 13
plt.rcParams.update({'font.size': fontsize})
plt.rcParams.update({'font.family': 'serif'})
fig = plt.figure(figsize=(7*scaling,
                          5*scaling))

ax0 = plt.subplot()

weights = np.ones_like(complete_histogram2[0:50000])/float(3*len(complete_histogram2[0:50000]))
weights = (weights,weights,weights)

ax0.grid()
n, bins, patches=ax0.hist((complete_histogram[0:50000],complete_histogram[50000:100000],complete_histogram[100000:150000]),bins=100, 
                          range=(0,0.25), weights=weights, color=("tab:green","tab:orange","tab:red"),stacked=True,histtype="bar",
                          label=("$n=1$","$n=2$","$n=3$"))


ax1=ax0.twinx()
weights = np.ones_like(complete_histogram2)/float(len(complete_histogram2))
n, bins, patches=ax1.hist(complete_histogram2,bins=1000, range=(0,0.251), weights=weights,cumulative=True,color="tab:blue",histtype='step')

ax0.set_xlabel("Normalized RMSE")
#plt.title("Histogram - Predicting Reaction Kinetics")
ax0.set_ylabel("Relative Frequency")
ax1.set_ylabel("Cumulative relative frequency")
ax0.set_xlim([0, 0.25])
ax0.set_ylim([0, 0.5])
ax1.set_ylim([0, 1.1])
ax0.set_yscale('symlog',linthreshy=0.01)

#fig.legend()
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig("cfh.png", dpi=320);
plt.savefig("cfh.pdf", dpi=320);
plt.show()
