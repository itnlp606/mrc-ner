# -*- coding:utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import pickle

def get_train_data():
	ddd = {}
	id2input = {}
	path = 'train_data'
	fileList = os.listdir(path)

	for idx, file in enumerate(fileList):
		name, suffix = os.path.splitext(file)
		name = int(name)
		entities = []
		with open(path+'/'+file, encoding='utf-8') as f:
			lines = f.read().splitlines()
			#print(lines)

			if suffix == '.txt':
				# print(lines[0])
				# cs += len(lines[0])

				if name in id2input:
					id2input[name][0] = lines[0]
				else:
					id2input[name] = [lines[0], None]
				
			else:
				for line in lines:
					entity = []
					s1 = line.split('\t')
					mid = s1[1].split(' ')
					entity.append(s1[0])
					entity.extend(mid)
					entity.append(s1[2])
					entities.append(entity)
					# sys.exit()
					if mid[0] in ddd:
						ddd[mid[0]] += 1
					else:
						ddd[mid[0]] = 1

					# dds += len(s1[2])
 
				if name in id2input:
					id2input[name][1] = entities
				else:
					id2input[name] = [None, np.array(entities)]

	for i in id2input:
		text, entities = id2input[i]
		for entity in entities:
			start, end, name = int(entity[2]), int(entity[3]), entity[4]
			if name != text[start:end]:
				print('wrong match')

	# print(dds)
	# print(cs)
	for i in ddd:
		print(i, ddd[i])
	return id2input

def load_data():
	with open('constants/data', 'rb') as f:
		content = pickle.load(f)
	return content

def get_train_dict():
	path = 'train_data'
	fileList = os.listdir(path)
	d = {}

	for idx, file in enumerate(fileList):
		name, suffix = os.path.splitext(file)
		name = int(name)
		with open(path+'/'+file, encoding='utf-8') as f:
			lines = f.read().splitlines()

			if suffix == '.ann':
				for line in lines:
					entity = []
					s1 = line.split('\t')
					mid = s1[1].split(' ')
					if s1[2] not in d:
						d[s1[2]] = mid[0]

	return d

if __name__ == '__main__':
	get_train_data()