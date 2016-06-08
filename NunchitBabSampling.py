import pandas as pd
import json
import sys

bab_csv = sys.argv[1] #"data/tg-bab-01.csv"
sample_csv = sys.argv[2] #"sample/tg-bab-sample-01.csv"
beforeInterval = int(sys.argv[3])

df = pd.read_csv(bab_csv, encoding='utf-8')

sampleTime = 10*1000 # 10sec
# beforeInterval = 60*1000 # 1min

stateType ={
	'unknown': 0,
	'attension': 1,
	'before': 2,
	'none': 3}
ringerMode = {'silent':1,'vibrate':2,'bell':3}
timeFeatures = ['accel','noti_posted','noti_removed', 'lock']
valueFeatures = ['ringer','battery','proximity','light']

sampleDataFrame = pd.DataFrame()
features = {}
status = {'time': df.timestamp[0],'state': -1, 'aheadUnlock': -1, 'lastUnlock': -1}
unlockData = df[df.type == 'unlock']

def initFeatures():
	global features
	features = {
		'lock': -1,				# lock time
		'unlock_duration':-1,	# last unlock~screen on duration
		'screen_on':0,			# num of screen on after last lock
		'ringer': 0,			# current ringer mode
		'battery': 0,			# current battery mode
		'proximity': -1, 		# current proximity
		'light': -1,			# current brightness
		'accel': -1,			# accel change time
		'noti_pendding': [],	# pendding noti
		'noti_posted': -1,		# noti post time
		'noti_removed': -1}		# noti remove time

def popUnlockData():
	global unlockData, status
	if len(unlockData) == 0:
		return False
	status['aheadUnlock'] = unlockData.timestamp.values[0]
	unlockData = unlockData[1:]
	if status['aheadUnlock'] <= status['time']:
		return popUnlockData()
	return True

def checkBeforeState():
	global stateType
	global beforeInterval
	global status
	if status['state'] != stateType['none']:
		return
	if status['aheadUnlock'] < 0:
		return
	if status['aheadUnlock'] - status['time'] <= beforeInterval:
		status['state'] = stateType['before']

def adaptFeatures(index, timestamp):
	global df
	global ringerMode
	global timeFeatures
	global valueFeatures
	global status, features, unlockData
	featureType = df.type[index]
	#if featureType == 'nunchitbab_start':
		# initFeatures()
		# aheadUnlockTime = -1
		# unlockedTime = time
	if featureType == 'unlock':
		status['lastUnlock'] = timestamp
		features['screen_on'] = 0
		popUnlockData()
		status['state'] = stateType['attension']
	elif featureType == 'screen':
		if json.loads(df.json[index])['onoff'] == 0: # off
			if status['state'] == stateType['attension']:
				features['lock'] = timestamp
				if status['lastUnlock'] > 0:
					features['unlock_duration'] = timestamp - status['lastUnlock']
					status['lastUnlock'] = -1
				status['state'] = stateType['none']
				checkBeforeState()
		else:
			features['screen_on'] += 1
	elif featureType in timeFeatures:
		features[featureType] = timestamp
		if featureType == 'noti_posted':
			if json.loads(df.json[index])['id'] not in features['noti_pendding']:
				features['noti_pendding'].append(json.loads(df.json[index])['id'])
		elif featureType == 'noti_removed':
			if json.loads(df.json[index])['id'] in features['noti_pendding']:
				features['noti_pendding'].remove(json.loads(df.json[index])['id'])
	elif featureType in valueFeatures:
		jsonParse = json.loads(df.json[index])
		for key in jsonParse:
			if featureType == 'ringer':
				features[featureType] = ringerMode[jsonParse[key]]
			else:
				features[featureType] = jsonParse[key]

def sampleData():
	global valueFeatures
	global timeFeatures
	global sampleDataFrame, features, status
	sampleData={}
	sampleData['timestamp'] = status['time']
	sampleData['state'] = status['state']
	for feature in valueFeatures:
		sampleData[feature] = features[feature]
	for feature in timeFeatures:
		if features[feature] < 0:
			sampleData[feature] = -1
		else:
			sampleData[feature] = status['time'] - features[feature]
	sampleData['unlock_duration'] = features['unlock_duration']
	sampleData['noti_pendding'] = len(features['noti_pendding'])
	sampleData['screen_on'] = features['screen_on']
	sampleDataFrame = sampleDataFrame.append(sampleData, ignore_index = True)

initFeatures()
popUnlockData()

print ("="*60)
print ('Sampling data from "' + bab_csv + '" (length:%d)' % len(df))
print ('Sampling time: %d' % sampleTime)
print ('before-state interval: %d' % beforeInterval)
print ("="*60)
index = 0
while index < len(df): # in range(20): # loop
	status['time'] = status['time'] + sampleTime
	while status['time'] >= df.timestamp[index]:
		adaptFeatures(index = index, timestamp = df.timestamp[index])
		index += 1
		if index % 10000 == 0:
			print (index)
		if index >= len(df):
			break
	if index >= len(df):
		break
	checkBeforeState()
	sampleData()

sampleDataFrame.to_csv(sample_csv, index = False, header = True)

print ('Sampling finished (Total %d samples)' % len(sampleDataFrame))
print ("="*60)
