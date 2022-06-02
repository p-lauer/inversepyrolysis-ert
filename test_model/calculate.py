#!/usr/bin/env python
# coding: utf-8

import time
import math
import numpy as np
import scipy
from scipy.signal import argrelextrema
from scipy.signal import peak_widths
from multiprocessing import Process
import multiprocessing
import pickle

import warnings
warnings.filterwarnings('ignore')

def _1gaussian(x_arr, peak1, cen1, sigma1):
    x_arr=np.array(x_arr)
    return peak1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x_arr - cen1) / sigma1) ** 2)))

def _2gaussian(x_arr, peak1, cen1, sigma1, peak2, cen2, sigma2):
    return _1gaussian(x_arr, peak1, cen1, sigma1)+_1gaussian(x_arr, peak2, cen2, sigma2)

def _3gaussian(x_arr, peak1, cen1, sigma1, peak2, cen2, sigma2, peak3, cen3, sigma3):
    return _1gaussian(x_arr, peak1, cen1, sigma1)+_1gaussian(x_arr, peak2, cen2, sigma2)+_1gaussian(x_arr, peak3, cen3, sigma3)

def _2components(frt, t):
    x_array = list(range(20, 551, 2))
    y_array_2gauss = frt[t * 266:266 + t * 266]
    try:
        if len(argrelextrema(frt[t * 266:266 + t * 266], np.greater)[0]) == 2:
            a1, a2 = argrelextrema(frt[t * 266:266 + t * 266], np.greater)[0]
            b1, b2 = peak_widths(frt[t * 266:266 + t * 266], rel_height=.9999, peaks=[a1, a2])[0]
            try:
                peak1, cen1, sigma1 = frt[t * 266 + a1], x_array[a1], b1 / 4
                peak2, cen2, sigma2 = frt[t * 266 + a2], x_array[a2], b2 / 4
                popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_2gaussian, x_array, y_array_2gauss,
                                                                    p0=[peak1, cen1, sigma1, peak2, cen2, sigma2],
                                                                    maxfev=500000, bounds=(0,np.inf))
            except:
                peak1, cen1, sigma1 = frt[t * 266 + a1], x_array[a1], b1 / 2
                peak2, cen2, sigma2 = frt[t * 266 + a2], x_array[a2], b2 / 2
                popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_2gaussian, x_array, y_array_2gauss,
                                                                    p0=[peak1, cen1, sigma1, peak2, cen2, sigma2],
                                                                    maxfev=500000, bounds=(0,np.inf))
            pars_1 = popt_2gauss[0:3]
            pars_2 = popt_2gauss[3:6]
            gauss_peak_1 = _1gaussian(x_array, *pars_1)
            gauss_peak_2 = _1gaussian(x_array, *pars_2)
            area1 = np.sqrt((np.trapz(gauss_peak_1, x_array)) ** 2)
            area2 = np.sqrt((np.trapz(gauss_peak_2, x_array)) ** 2)
        else:
            a1 = argrelextrema(frt[t * 266:266 + t * 266], np.greater)[0][0]
            b1 = peak_widths(frt[t * 266:266 + t * 266], rel_height=.9999, peaks=[a1])[0][0] + 1
            try:
                peak1, cen1, sigma1 = frt[t * 266 + a1], x_array[a1], b1 / 4
                popt_1gauss, pcov_1gauss = scipy.optimize.curve_fit(_1gaussian, x_array, y_array_2gauss,
                                                                    p0=[peak1, cen1, sigma1], maxfev=500000, bounds=(0,np.inf))
            except:
                peak1, cen1, sigma1 = frt[t * 266 + a1], x_array[a1], b1 / 2
                popt_1gauss, pcov_1gauss = scipy.optimize.curve_fit(_1gaussian, x_array, y_array_2gauss,
                                                                    p0=[peak1, cen1, sigma1], maxfev=500000, bounds=(0,np.inf))
            gauss_peak_1 = _1gaussian(x_array, *popt_1gauss)
            area1 = np.sqrt(np.trapz(gauss_peak_1, x_array) ** 2)
            area2 = np.sqrt((np.trapz(y_array_2gauss, x_array) - area1) ** 2)
    except:
        return [0.5, 0.5]
    return sorted([area1 / (area1 + area2), area2 / (area1 + area2)])


def _3components(frt, t):
    x_array = list(range(20, 551, 2))
    y_array_3gauss = frt[t * 266:266 + t * 266]
    try:
        if len(argrelextrema(frt[t * 266:266 + t * 266], np.greater)[0]) == 3:
            a1, a2, a3 = argrelextrema(frt[t * 266:266 + t * 266], np.greater)[0]
            b1, b2, b3 = peak_widths(frt[t * 266:266 + t * 266], rel_height=.9999, peaks=[a1, a2, a3])[0]
            try:
                peak1, cen1, sigma1 = frt[t * 266 + a1], x_array[a1], b1 / 4
                peak2, cen2, sigma2 = frt[t * 266 + a2], x_array[a2], b2 / 4
                peak3, cen3, sigma3 = frt[t * 266 + a3], x_array[a3], b3 / 4
                popt_3gauss, pcov_3gauss = scipy.optimize.curve_fit(_3gaussian, x_array, y_array_3gauss,
                                                                    p0=[peak1, cen1, sigma1, peak2, cen2, sigma2, peak3,
                                                                        cen3, sigma3], maxfev=500000, bounds=(0,np.inf))
            except:
                peak1, cen1, sigma1 = frt[t * 266 + a1], x_array[a1], b1 / 2
                peak2, cen2, sigma2 = frt[t * 266 + a2], x_array[a2], b2 / 2
                peak3, cen3, sigma3 = frt[t * 266 + a3], x_array[a3], b3 / 2
                popt_3gauss, pcov_3gauss = scipy.optimize.curve_fit(_3gaussian, x_array, y_array_3gauss,
                                                                    p0=[peak1, cen1, sigma1, peak2, cen2, sigma2, peak3,
                                                                        cen3, sigma3], maxfev=500000, bounds=(0,np.inf))
            pars_1 = popt_3gauss[0:3]
            pars_2 = popt_3gauss[3:6]
            pars_3 = popt_3gauss[6:9]
            gauss_peak_1 = _1gaussian(x_array, *pars_1)
            gauss_peak_2 = _1gaussian(x_array, *pars_2)
            gauss_peak_3 = _1gaussian(x_array, *pars_3)
            area1 = np.sqrt((np.trapz(gauss_peak_1, x_array)) ** 2)
            area2 = np.sqrt((np.trapz(gauss_peak_2, x_array)) ** 2)
            area3 = np.sqrt((np.trapz(gauss_peak_3, x_array)) ** 2)
        else:
            a1, a2 = argrelextrema(frt[t * 266:266 + t * 266], np.greater)[0]
            b1, b2 = peak_widths(frt[t * 266:266 + t * 266], rel_height=.9999, peaks=[a1, a2])[0]
            try:
                peak1, cen1, sigma1 = frt[t * 266 + a1], x_array[a1], b1 / 4,
                peak2, cen2, sigma2 = frt[t * 266 + a2], x_array[a2], b2 / 4
                popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_2gaussian, x_array, y_array_3gauss,
                                                                    p0=[peak1, cen1, sigma1, peak2, cen2, sigma2],
                                                                    maxfev=500000, bounds=(0,np.inf))
            except:
                peak1, cen1, sigma1 = frt[t * 266 + a1], x_array[a1], b1 / 2,
                peak2, cen2, sigma2 = frt[t * 266 + a2], x_array[a2], b2 / 2
                popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_2gaussian, x_array, y_array_3gauss,
                                                                    p0=[peak1, cen1, sigma1, peak2, cen2, sigma2],
                                                                    maxfev=500000, bounds=(0,np.inf))
            pars_1 = popt_2gauss[0:3]
            pars_2 = popt_2gauss[3:6]
            gauss_peak_1 = _1gaussian(x_array, *pars_1)
            gauss_peak_2 = _1gaussian(x_array, *pars_2)
            area1 = np.sqrt((np.trapz(gauss_peak_1, x_array)) ** 2)
            area2 = np.sqrt((np.trapz(gauss_peak_2, x_array)) ** 2)
            area3 = np.sqrt((np.trapz(y_array_3gauss, x_array) - area1 - area2) ** 2)
    except:
        return [0.35, 0.35, 0.3]
    return sorted([area1 / (area1 + area2 + area3), area2 / (area1 + area2 + area3), area3 / (area1 + area2 + area3)])


def predictnumbers(ftz):
    n = moda.predict([ftz])
    try:
        if n == 1:
            reslt = np.concatenate((modc1.predict([np.append(ftz, 0)])[0], [1, 0, 0]))
            reslt = np.insert(reslt, [1,1,2,2], 0)
        elif n == 2:
            ex = np.zeros((4, 2))
            for jt, teach in enumerate(ex):
                ex[jt] = np.sort(_2components(ftz, jt))
            reslt = np.concatenate(
                (modc2.predict([np.concatenate((ftz, np.nanmean(ex, axis=0)))])[0], np.nanmean(ex, axis=0)))
            reslt = np.insert(reslt, [2,4,6], 0)
        elif n == 3:
            ex = np.zeros((4, 3))
            for jt, teach in enumerate(ex):
                ex[jt] = np.sort(_3components(ftz, jt))
            reslt = np.concatenate(
                (modc3.predict([np.concatenate((ftz, np.nanmean(ex, axis=0)))])[0], np.nanmean(ex, axis=0)))
    except:
        reslt = np.zeros(9)
    return reslt


def multipredict(datazet, nr):
    datazet = np.array(datazet)
    for ik, each in enumerate(datazet):
        return_list[len(datazet) * nr + ik] = predictnumbers(each)


cores = 100
samplesize = 150000  # NUMBER OF SAMPLES
samplesperset = 266

firstmodel1 = "../build_model/sm1.pickle"
thirdmodel1 = "../build_model/sm3_1r.pickle"
thirdmodel2 = "../build_model/sm3_2r.pickle"
thirdmodel3 = "../build_model/sm3_3r.pickle"

print('Loading dataset')
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

print('Dataset loaded')

print('Loading models')
with open(thirdmodel2, 'rb') as file:
    modc2 = pickle.load(file)
with open(thirdmodel3, 'rb') as file:
    modc3 = pickle.load(file)
with open(firstmodel1, 'rb') as file:
    moda = pickle.load(file)
with open(thirdmodel1, 'rb') as file:
    modc1 = pickle.load(file)

print('Models loaded')

k = samplesize
eval_predictions = features[:k]

starttime = time.time()
if __name__ == '__main__':
    processes = []
    manager = multiprocessing.Manager()
    return_list = manager.list([0] * k)
    for n in range(cores):
        p = Process(target=multipredict,
                    args=(eval_predictions[int((n * (k / cores))):int(((k / cores) + n * (k / cores)))], n,))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
endtime = time.time()
duration = endtime - starttime
print("Calculation Duration: {}s".format(duration))
print(" ")
prediction = np.array(return_list)

np.savetxt("prediction.csv", prediction, delimiter=",")

print('Prediction finished')
