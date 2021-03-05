import sys
import random
from sklearn.svm import LinearSVC

trainX, trainY = [], []
t_index = []
testX = []
trueY = []
newtestX = []
newtrainX = []


datafile = sys.argv[1]
file = open(datafile)
data = []
i = 0
l = file.readline()
while(l != '') : 
    a = l.split()
    t = len(a)
    l2 = []
    for j in range(0, t, 1):
        l2.append(float(a[j]))
        if j == (t-1) :
            l2.append(float(1))
    data.append(l2)
    l = file.readline()
rows = len(data)
cols = len(data[0])
file.close()


labelFile = sys.argv[2]
trainlabels = open(labelFile, encoding='utf-8').readlines()
trainlabels = [line.strip().split(' ') for line in trainlabels]
trainlabels = [list(map(int, line)) for line in trainlabels]


def getbestC(train,labels):            
    random.seed()
    allCs = [.001, .01, .1, 1, 10, 100]
    error = {}
    for j in range(0, len(allCs), 1):
            error[allCs[j]] = 0
    rowIDs = []
    for i in range(0, len(train), 1):
            rowIDs.append(i)
    num_splits = 10
    for x in range(0,num_splits,1):        
            Ntrain = []
            Nlabels = []
            valid = []
            validlabels = []
            random.shuffle(rowIDs)    
            for i in range(0, int(.9*len(rowIDs)), 1):
                    Ntrain.append(train[i])
                    Nlabels.append(labels[i])
            for i in range(int(.9*len(rowIDs)), len(rowIDs), 1):
                    valid.append(train[i])
                    validlabels.append(labels[i])
            for j in range(0, len(allCs), 1):
                    C = allCs[j]
                    clf = LinearSVC(C=C)
                    clf.fit(Ntrain, Nlabels)
                    predict = clf.predict(valid)
                    err = 0
                    for i in range(0, len(predict), 1):
                            if(predict[i] != validlabels[i]):
                                    err = err + 1
                    err = err/len(validlabels)
                    error[C]+=err
                    
    bestC = 0
    min_err=100
    keys = list(error.keys())
    for i in range(0, len(keys), 1):
           key = keys[i]
           error[key] = error[key]/num_splits
           if(error[key] < min_err):
                   min_err = error[key]
                   bestC = key
    
    return [bestC,min_err]

def running_it(tX, tY, testX):     
    best_C, min_error = getbestC(tX, tY)
    clf = LinearSVC(C = best_C, max_iter=10000)
    clf.fit(tX, tY)
    pre = clf.predict(testX)
    for i, pred in enumerate(pre):
        if pred == -1:
            pre[i] = 0
    return pre


for (label, i) in trainlabels:
    trainX.append(data[i])
    if label == 1:
        trainY.append(1)
    elif label == 0:
        trainY.append(-1)
    t_index.append(i)
TI = [i for i in range(len(data)) if i not in t_index]
if len(TI) == 0:
    TI = t_index
else:
    for i in TI:
        testX.append(data[i])
if len(testX) == 0:
    testX = trainX
predicts = running_it(trainX, trainY, testX)
for temp, index in zip(predicts, TI):
    print(temp, index)



