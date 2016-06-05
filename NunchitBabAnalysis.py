import pandas as pd
import sys
import math
import time

startTime = time.time()

sampledBab_csv = sys.argv[1]
df = pd.read_csv(sampledBab_csv)

depthLimit = int(sys.argv[2])

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

def entropy(dataFrame):
	total = float(len(dataFrame[dataFrame.state >= 0]))
	if total == 0:
		return 0
	probability = [
		len(dataFrame[dataFrame.state == 2])/total,
		len(dataFrame[dataFrame.state == 3])/total]
	if probability[0]+probability[1] != 1:
		print ('Error!: Sum of probability is not 1')
	# print ('Probabilities:', end=' ')
	# print (probability)
	entropy = 0
	for i in range(2):
		if probability[i] == 0:
			continue
		entropy -= probability[i]*math.log(probability[i],2)
	return entropy

def getLog(x):
	if x == 0:
		return math.log(0.1,10)
	else:
		return math.log(x,10)

featureType = []
for feature in df.columns:
	if feature != 'state' and feature != 'timestamp':
		featureType.append(feature)

class NunchiTree():
	def __init__(self, depth):
		self.subTree = []
		self.threshold = []
		self.depth = depth + 1
	def train(self, dataFrame):
		global featureType
		global depthLimit
		if entropy(dataFrame) < 0.05 or len(dataFrame) <= 10 or self.depth > depthLimit:
			if len(dataFrame[dataFrame.state == 2]) > len(dataFrame[dataFrame.state == 3]):
				self.result = 2
			else:
				self.result = 3
			# print('End node (depth: %d)' % self.depth)
			return
		bestInfoGain = -100
		targetFeature = ''
		targetThreshold = 0
		for feature in featureType:
			threshold = dataFrame[feature].min()
			delta = (dataFrame[feature].max() - dataFrame[feature].min())/10
			maxInfoGain = -100
			maxThreshold = 0
			for d in range(1,10):
				threshold += delta
				if len(dataFrame[dataFrame[feature] < threshold]) == 0 or len(dataFrame[dataFrame[feature] >= threshold]) == 0:
					continue
				infoGain = - entropy(dataFrame[dataFrame[feature] < threshold]) - entropy(dataFrame[dataFrame[feature] >= threshold])
				if maxInfoGain < infoGain:
					maxInfoGain = infoGain
					maxThreshold = threshold
			if bestInfoGain < maxInfoGain:
				bestInfoGain = maxInfoGain
				targetFeature = feature
				targetThreshold = maxThreshold
		self.targetFeature = targetFeature
		self.threshold = targetThreshold
		subTree0 = NunchiTree(self.depth)
		subTree1 = NunchiTree(self.depth)
		# print ('-'*5)
		# print ('target: '+targetFeature+', threshold: %f' % targetThreshold)
		# print ('[{0} / {1}]'.format(len(dataFrame[dataFrame[targetFeature] < targetThreshold]), len(dataFrame[dataFrame[targetFeature] >= targetThreshold])))
		subTree0.train(dataFrame[dataFrame[targetFeature] < targetThreshold])
		subTree1.train(dataFrame[dataFrame[targetFeature] >= targetThreshold])
		self.subTree = [subTree0,subTree1]
		return
	def getSubTree(self, features):
		if features[self.targetFeature] < self.threshold:
			return self.subTree[0]
		else:
			return self.subTree[1]
	def classify(self, features):
		if len(self.subTree) == 0:
			return self.result
		if features[self.targetFeature] < self.threshold:
			return self.subTree[0].classify(features)
		else:
			return self.subTree[1].classify(features)

def evaluateTree(tree, testSet):
	# print ('=== Evaluation with test set (size: %d)' % len(testSet))
	truePositive = 0
	trueNegative = 0
	falsePositive = 0
	falseNegative = 0
	invalid = 0
	total = 0
	for index in range(len(testSet)):
		result = tree.classify(testSet.loc[index])
		if result == testSet.loc[index]['state']:
			if result == 2:
				truePositive += 1
			elif result == 3:
				trueNegative += 1
			else:
				akward += 1
		else:
			if result == 2:
				falsePositive += 1
			elif result == 3:
				falseNegative += 1
			else:
				invalid +=1
		total += 1

	print ('[TP: {0}, TN: {1}, FP: {2}, FN: {3}]'.format(truePositive, trueNegative, falsePositive, falseNegative))
	print ('--- Recall(TP/TP+FN): %f' % (truePositive/(truePositive + falseNegative)))
	print ('--- Precision(TP/TP+FP): %f' % (truePositive/(truePositive + falsePositive)))
	if invalid > 0:
		print ('Invalid: %d' % invalid)

def validation(trainSet, testSet):
	print ('Training with train set (size: %d)' % len(trainDataFrame))
	tree = NunchiTree(0)
	tree.train(trainSet)
	# print ('Training finish.')
	evaluateTree(tree,testSet)

df = getClearDataFrame(df)
df['light'] = df['light'].apply(lambda x: getLog(x)) # apply log for light feature
df = df[df.state > 1]

trainBound1 = int(len(df)/3)
trainBound2 = int(len(df)*2/3)

# trainDataFrame = df[0::2]
# testDataFrame = df[1::2]
# testDataFrame.index = range(len(testDataFrame))

print ("="*70)
print ('NunchitBab Analysis from "' + sampledBab_csv + '" (length:%d)' % len(df))
print ('Depth limit: %d' % depthLimit)
print ("="*70)
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
print ('\n(execution time: %d sec)' %(time.time() - startTime))
print ("="*70)