import pandas as pd
import sys
import math
import time
from sklearn import tree

startTime = time.time()

sampledBab_csv = sys.argv[1]
df = pd.read_csv(sampledBab_csv)

def getClearDataFrame(dataFrame):
	index = 0
	while True:
		hasInvalid = False
		for c in dataFrame.columns:
			if dataFrame[c][index] < 0:
				hasInvalid = True
		if not hasInvalid:
			break
		index += 1
	return dataFrame[index:]

def getLog(x):
	if x == 0:
		return math.log(0.1,10)
	else:
		return math.log(x,10)

def validation(trainSet, testSet):
	clf = tree.DecisionTreeClassifier()
	trainX = trainSet.drop('state', axis=1).drop('timestamp', axis=1).values
	trainY = trainSet['state'].values
	clf = clf.fit(trainX, trainY)
	testX = testSet.drop('state', axis=1).drop('timestamp', axis=1).values
	testY = testSet['state'].values

	predictY = clf.predict(testX)

	truePositive = 0
	trueNegative = 0
	falsePositive = 0
	falseNegative = 0
	invalid = 0
	total = 0
	for index in range(len(testY)):
		if predictY[index] == testY[index]:
			if predictY[index] == 2:
				truePositive += 1
			elif predictY[index] == 3:
				trueNegative += 1
			else:
				akward += 1
		else:
			if predictY[index] == 2:
				falsePositive += 1
			elif predictY[index] == 3:
				falseNegative += 1
			else:
				invalid +=1
		total += 1

	print ('[TP: {0}, TN: {1}, FP: {2}, FN: {3}]'.format(truePositive, trueNegative, falsePositive, falseNegative))
	print ('--- Recall(TP/TP+FN): %f' % (truePositive/(truePositive + falseNegative)))
	print ('--- Precision(TP/TP+FP): %f' % (truePositive/(truePositive + falsePositive)))
	if invalid > 0:
		print ('Invalid: %d' % invalid)

df = getClearDataFrame(df)
df['light'] = df['light'].apply(lambda x: getLog(x)) # apply log for light feature
df = df[df.state > 1]

trainBound1 = int(len(df)/3)
trainBound2 = int(len(df)*2/3)

print ("="*75)
print ('Nunchi Decision Tree from "' + sampledBab_csv + '" (length:%d)' % len(df))
print ("="*75)
print ('[Train: [{0}:{1}] | Test: [{1}:{2}]'.format(0,trainBound2,len(df)))
trainDataFrame = df[0 : trainBound2]
testDataFrame = df[trainBound2 : ]
testDataFrame.index = range(len(testDataFrame))
validation(trainDataFrame, testDataFrame)
print ('\n[Train: [{0}:{1},{2}:{3}] | Test: [{1}:{2}]'.format(0,trainBound1,trainBound2,len(df)))
trainDataFrame = df[0 : trainBound1].append(df[trainBound2:])
testDataFrame = df[trainBound1 : trainBound2]
testDataFrame.index = range(len(testDataFrame))
validation(trainDataFrame, testDataFrame)
print ('\n[Train: [{1}:{2}] | Test: [{0}:{1}]'.format(0,trainBound1,len(df)))
trainDataFrame = df[trainBound1 : ]
testDataFrame = df[0 : trainBound1]
testDataFrame.index = range(len(testDataFrame))
validation(trainDataFrame, testDataFrame)