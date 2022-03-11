#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
import numpy as np
import time
import itertools
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle

# INPUT VARIABLES

algorithm = "EFT"  # NAME OF ALGORITHM, as string
estimators = [200, ]  # SET OF NUMBER OF ESTIMATORS
max_depth = [None, ]  # SET OF maximum depth of a tree. If None, nodes are expanded until all leaves are pure
heatingrates = [5, 10, 30, 40]  # HEATING RATES TO BE EVALUATED
bootstrap = [False, ]

samplesize = 6000000  # NUMBER OF SAMPLES
samplesperset = 266
reactions = 2  # NUMBER OF REACTIONS IN SAMPLE

cores = 128
head = samplesize

# PREPROCESSING
labels = []
features = []

statistic = []
counter = 1
hrr = list(range(len(heatingrates)))
# hrrlist=list(itertools.chain.from_iterable(itertools.combinations(hrr, r) for r in range(1,len(hrr)+1)))
hrrlist = [(0, 1, 2, 3)]
gridlist = list(itertools.product(estimators, max_depth, hrrlist, bootstrap))

currenttime = time.time()
print(str(head))

print('Loading Dataset....')

for i in range(120):
    feat = "../generate_db/features6400k_5_10_30_40_2r_2ks_" + str(i) + ".csv"
    lab = "../generate_db/labels6400k_5_10_30_40_2r_2ks_" + str(i) + ".csv"

    labels_temp = np.genfromtxt(lab, delimiter=',', dtype=np.float64)[:, 9:]
    features_temp = np.genfromtxt(feat, delimiter=',', dtype=np.float64)
    labels_temp[:, 0] = np.log(labels_temp[:, 0])
    labels_temp[:, 1] = np.log(labels_temp[:, 1])

    features_temp = np.hstack((features_temp[:, 0:samplesperset * 4], labels_temp[:, 6:reactions + 6]))
    labels_temp = np.hstack((labels_temp[:, 0:reactions], labels_temp[:, 3:reactions + 3]))

    features.append(features_temp)
    labels.append(labels_temp)

features = np.array(features).reshape(samplesize, -1)
labels = np.array(labels).reshape(samplesize, -1)

train_features_tot, test_features_tot, train_labels_tot, test_labels_tot = train_test_split(features, labels,
                                                                                            test_size=0.25,
                                                                                            random_state=42)

print('Training Features Shape:', train_features_tot.shape)
print('Training Labels Shape:', train_labels_tot.shape)
print('Testing Features Shape:', test_features_tot.shape)
print('Testing Labels Shape:', test_labels_tot.shape)

eval_features = test_features_tot[:, :-reactions]

train_labels = train_labels_tot
test_labels = test_labels_tot

train_features_all = []
test_features_all = []

for i in hrr:
    train_features_all.append(train_features_tot[:, 0 + i * samplesperset:(i + 1) * samplesperset])
    test_features_all.append(test_features_tot[:, 0 + i * samplesperset:(i + 1) * samplesperset])
train_features_all.append(train_features_tot[:, len(hrr) * samplesperset:])
test_features_all.append(test_features_tot[:, len(hrr) * samplesperset:])

print('Dataset loaded')

# CALCULATION LOOP

for i in gridlist:
    print(
        "case_{}_{}k_{}r_{}HR_{}est_{}sl_BS{}".format(algorithm, samplesize / 1000, reactions, i[2], i[0], i[1], i[3]))
    train_features = np.hstack(list(train_features_all[t] for t in i[2]))
    test_features = np.hstack(list(test_features_all[t] for t in i[2]))
    train_features = np.hstack((train_features, train_features_all[-1]))
    test_features = np.hstack((test_features, test_features_all[-1]))
    train_features = np.array(train_features)
    clf = ExtraTreesRegressor(n_estimators=i[0], max_depth=i[1], bootstrap=i[3], n_jobs=cores, random_state=2)
    print("Start fitting; calculation ", str(counter), " of ", str(len(gridlist)))
    counter += 1
    currenttime = time.time()
    clf.fit(train_features, train_labels);
    out_file = open('sm3_2r.pickle', 'wb')
    pickle.dump(clf, out_file)
    out_file.close()
    print("Fitting time: ", (time.time() - currenttime) / 60, ' min')
    predictions = clf.predict(test_features)
    #np.savetxt(
    #    "test_labels_{}_{}k_{}r_{}HR_{}est_{}sl_BS{}.csv".format(algorithm, samplesize / 1000, reactions, i[2], i[0],
    #                                                             i[1], i[3]), test_labels, delimiter=",")
    #np.savetxt(
    #    "test_predictions_{}_{}k_{}r_{}HR_{}est_{}sl_BS{}.csv".format(algorithm, samplesize / 1000, reactions, i[2],
    #                                                                  i[0], i[1], i[3]), predictions, delimiter=",")
    #np.savetxt(
    #    "fractions_{}_{}k_{}r_{}HR_{}est_{}sl_BS{}.csv".format(algorithm, samplesize / 1000, reactions, i[2], i[0],
    #                                                           i[1], i[3]), test_features[:, -reactions:],
    #    delimiter=",")
    print("R2 score: {}".format(r2_score(test_labels, predictions)))
    print("MSE: {}".format(mean_squared_error(test_labels, predictions)))
    print('Total time: {}'.format(time.time() - currenttime))


