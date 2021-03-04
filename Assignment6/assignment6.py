import sys

from gini import getTheGinie

################ DEFINING HYPERPARAMETERS AND FILE PATHS ################

'''dataFile = sys.argv[1]
labelFile = sys.argv[2]
saveFile = 'results.0'
'''

################ DATA PRE-PROCESSING ################

data = open("/Users/pavanghuge/Downloads/ML/Test Data/ionosphere/ionosphere.data")
data = [line.strip().split() for line in data]
data = [list(map(float, line)) for line in data]

trainlabels = open("/Users/pavanghuge/Downloads/ML/Test Data/ionosphere/ionosphere.trainlabels.0")
trainlabels = [line.strip().split(' ') for line in trainlabels]
trainlabels = [list(map(int, line)) for line in trainlabels]

trainX, trainY = [], []
trainIndex = []

for (label, i) in trainlabels:
    trainX.append(data[i])

    if label == 1:
        trainY.append(1)
    else:
        trainY.append(-1)

    trainIndex.append(i)

testX = []
testIndex = [i for i in range(len(data)) if i not in trainIndex]

if len(testIndex) == 0:
    testIndex = trainIndex

else:
    for i in testIndex:
        testX.append(data[i])

if len(testX) == 0:
    testX = trainX

################ TRAINING MODEL ################

myGinie = getTheGinie()
wish = myGinie.grantMyWish(trainX, trainY)

for (k, s) in wish:
    print(k, s)