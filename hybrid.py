import surprise
import surprise.accuracy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

data = surprise.Dataset.load_builtin('ml-1m')

kSplit = surprise.model_selection.KFold(n_splits=5, shuffle=True)

sim_options = {
    'name': 'pearson_baseline',
    'user_based': True
}
collabKNN = surprise.KNNBasic(k=40, sim_options=sim_options)
rmseKNN = []

slopeOne = surprise.prediction_algorithms.slope_one.SlopeOne()
rmseSlope = []


class HybridPrediction(surprise.AlgoBase):
    def __init__(self):
        surprise.AlgoBase.__init__(self)

    def fit(self, trainset):
        surprise.AlgoBase.fit(self, trainset)
        return self

    def get_val(self, est):
        if isinstance(est, tuple):
            return est[0]
        return est

    def estimate(self, u, i):
        # print(collabKNN.estimate(u, i))
        # print(slopeOne.estimate(u, i))
        v1 = self.get_val(collabKNN.estimate(u, i))
        v2 = self.get_val(slopeOne.estimate(u, i))
        return v1 * 0.3 + v2 * 0.7


hybrid = HybridPrediction()
rmseHybrid = []

for trainset, testset in kSplit.split(data):
    collabKNN.fit(trainset)
    predictionsKNN = collabKNN.test(testset)
    rmseKNN.append(surprise.accuracy.rmse(predictionsKNN))

    slopeOne.fit(trainset)
    predSlope = slopeOne.test(testset)
    rmseSlope.append(surprise.accuracy.rmse(predSlope))

    hybrid.fit(trainset)
    predHybrid = hybrid.test(testset)
    rmseHybrid.append(surprise.accuracy.rmse(predHybrid))

print('KNN results:')
print(rmseKNN)

print('SlopeOne results:')
print(rmseSlope)

print('Hybrid results:')
print(rmseHybrid)
