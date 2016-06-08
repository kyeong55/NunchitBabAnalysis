import pandas as pd
import numpy as np
import sys
import math

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

def getVectorizedDataFrame(dataFrame):
	# battery: 2,3,5
	dataFrame['battery_charging'] = dataFrame['battery'].apply(lambda x: 0 + (x == 2))
	dataFrame['battery_discharging'] = dataFrame['battery'].apply(lambda x: 0 + (x == 3))
	dataFrame['battery_full'] = dataFrame['battery'].apply(lambda x: 0 + (x == 5))
	# proximity: 0,5
	dataFrame['proximity_near'] = dataFrame['proximity'].apply(lambda x: 0 + (x == 0))
	dataFrame['proximity_far'] = dataFrame['proximity'].apply(lambda x: 0 + (x == 5))
	# ringer: 1,2,3
	dataFrame['ringer_silent'] = dataFrame['ringer'].apply(lambda x: 0 + (x == 1))
	dataFrame['ringer_vibrate'] = dataFrame['ringer'].apply(lambda x: 0 + (x == 2))
	dataFrame['ringer_bell'] = dataFrame['ringer'].apply(lambda x: 0 + (x == 3))

	return dataFrame.drop('battery', axis=1).drop('proximity', axis=1).drop('ringer', axis=1)

size_of_layers = []
layers =[]
weights = []
learning_rate = 1

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def initLayers():
	global size_of_layers
	global layers
	global weights
	np.random.seed(1)
	layers =[]
	weights = []
	for i in range(len(size_of_layers)):
		layers.append(np.ones((1, size_of_layers[i])))
	# layers =[
	#     np.ones((1, size_of_layers[0])), # input layer 
	#     np.ones((1, size_of_layers[1])), # hidden layer
	#     np.ones((1, size_of_layers[2])), # hidden layer
	#     np.ones((1, size_of_layers[3])), # output layer
	# ]
	for i in range(1,len(size_of_layers)):
		weights.append((2 * np.random.random((size_of_layers[i-1], size_of_layers[i]))) - 1)
	# weights = [
	#     (2 * np.random.random((size_of_layers[0], size_of_layers[1]))) - 1, # input layer - hidden layer
	#     (2 * np.random.random((size_of_layers[1], size_of_layers[2]))) - 1, # hidden layer - hidden layer
	#     (2 * np.random.random((size_of_layers[2], size_of_layers[3]))) - 1, # hidden layer - output layer
	# ]

def propagate(x):
	global layers
	layers[0] = x
	for i in range(1,len(layers)):
		layers[i] = np.dot(layers[i-1],weights[i-1])
		layers[i] = sigmoid(layers[i])
	return layers[len(layers) - 1]

def backpropagate(y_true, learning_rate):
	global size_of_layers
	global layers
	global weights
	size = len(weights)
	delta = [1]*size
	delta[size-1] = layers[size]*(np.ones((1,size_of_layers[size])) - layers[size])*(y_true - layers[size])
	for i in range(1,size):
		delta[size-i-1] = layers[size-i]*(np.ones((1,size_of_layers[size-i])) - layers[size-i])*np.dot(delta[size-i],weights[size-i].transpose())

	# delta3 = layers[3]*(np.ones((1,size_of_layers[3])) - layers[3])*(y_true - layers[3])
	# delta2 = layers[2]*(np.ones((1,size_of_layers[2])) - layers[2])*np.dot(delta3,weights[2].transpose())
	# delta1 = layers[1]*(np.ones((1,size_of_layers[1])) - layers[1])*np.dot(delta2,weights[1].transpose())
	for j in range(size):
		i = size - j - 1
		weights[i] = weights[i] + learning_rate*np.dot(layers[i].transpose(),delta[i])
	# weights[2] = weights[2] + learning_rate*np.dot(layers[2].transpose(),delta3)
	# weights[1] = weights[1] + learning_rate*np.dot(layers[1].transpose(),delta2)
	# weights[0] = weights[0] + learning_rate*np.dot(layers[0].transpose(),delta1)

# def evaluate(X, Y):
#     error = []
#     for i in range(0, len(X)):
#         x =  X[i, np.newaxis]
#         y =  Y[i, np.newaxis]

#         y_predicted = propagate(x)

#         error.append(np.mean(np.abs(y_predicted - y)))
    
#     return np.mean(error)

def validation(trainSet, testSet):
	global learning_rate

	initLayers()

	trainX = trainSet.drop('state', axis=1).drop('timestamp', axis=1).values
	trainY = pd.DataFrame()
	trainY['state_2'] = trainSet['state'].apply(lambda x: 0 + (x == 2))
	trainY['state_3'] = trainSet['state'].apply(lambda x: 0 + (x == 3))
	testX = testSet.drop('state', axis=1).drop('timestamp', axis=1).values
	testY = pd.DataFrame()
	testY['state_2'] = testSet['state'].apply(lambda x: 0 + (x == 2))
	testY['state_3'] = testSet['state'].apply(lambda x: 0 + (x == 3))

	train_X = np.array(trainX)
	train_Y = np.array(trainY)
	test_X = np.array(testX)
	test_Y = np.array(testY)

	for index in range(len(train_Y)):
		propagate(train_X[index,np.newaxis])
		backpropagate(train_Y[index,np.newaxis], learning_rate)
		# if (index % 100) == 0:
		# 	print ('training progress: {0}/{1}'.format(index,len(train_Y)))
	# print ('Training finish.')
	# print ('[Evaluation]')

	truePositive = 0
	trueNegative = 0
	falsePositive = 0
	falseNegative = 0
	total = 0
	for index in range(len(test_Y)):
		predictY = propagate(test_X[index]).argmax()
		trueY = test_Y[index].argmax()

		if predictY == trueY:
			if predictY == 0:
				truePositive += 1
			else:
				trueNegative += 1
		else:
			if predictY == 0:
				falsePositive += 1
			else:
				falseNegative += 1
		total += 1

	print ('[TP: {0}, TN: {1}, FP: {2}, FN: {3}]'.format(truePositive, trueNegative, falsePositive, falseNegative))
	if (truePositive + falseNegative) > 0:
		print ('--- Recall(TP/TP+FN): %f' % (truePositive/(truePositive + falseNegative)))
	if (truePositive + falsePositive) > 0:
		print ('--- Precision(TP/TP+FP): %f' % (truePositive/(truePositive + falsePositive)))

sampledBab_csv = sys.argv[1]
df = pd.read_csv(sampledBab_csv)

df = getClearDataFrame(df)
df['light'] = df['light'].apply(lambda x: getLog(x)) # apply log for light feature
df = df[df.state > 1]
df = getVectorizedDataFrame(df)

size_of_layers = [
    len(df.columns) - 2,
    1000,
    2,
]

trainBound1 = int(len(df)/3)
trainBound2 = int(len(df)*2/3)

print ("="*75)
print ('Nunchi Neural Network from "' + sampledBab_csv + '" (length:%d)' % len(df))
print ("="*75)


print ('\nTrain: [{0}:{1}] | Test: [{1}:{2}]'.format(0,trainBound2,len(df)))
trainDataFrame = df[0 : trainBound2]
trainDataFrame.index = range(len(trainDataFrame))
testDataFrame = df[trainBound2 : ]
testDataFrame.index = range(len(testDataFrame))
validation(trainDataFrame, testDataFrame)

print ('\nTrain: [{0}:{1},{2}:{3}] | Test: [{1}:{2}]'.format(0,trainBound1,trainBound2,len(df)))
trainDataFrame = df[0 : trainBound1].append(df[trainBound2:])
trainDataFrame.index = range(len(trainDataFrame))
testDataFrame = df[trainBound1 : trainBound2]
testDataFrame.index = range(len(testDataFrame))
validation(trainDataFrame, testDataFrame)

print ('\nTrain: [{1}:{2}] | Test: [{0}:{1}]'.format(0,trainBound1,len(df)))
trainDataFrame = df[trainBound1 : ]
trainDataFrame.index = range(len(trainDataFrame))
testDataFrame = df[0 : trainBound1]
testDataFrame.index = range(len(testDataFrame))
validation(trainDataFrame, testDataFrame)