import re
import sys
import torch
import pickle
import numpy as np
from constants import TAG2ID, LABEL2Q

# merge ques, data and pad this batch
def preprocessing(data, tokenizer):
	# init ds
	space = ','
	data_to_be_padded = []
	tags_to_be_padded = []
	followed = []
	labels_len = []

	for kdx, tup in enumerate(data):
		# For each paragarph, generate tensors, Labels
		data, labels = tup

		## process tags
		# init
		label2tag = {}
		for name in LABEL2Q:
			label2tag[name] = [TAG2ID['N']]*len(data)

		# add tagging info
		for tup in labels:
			_, label, start, end, name = tup
			start, end = int(start), int(end)
			# make sure span and name match
			if data[start:end] != name:
				print('wrong dataset')
			label2tag[label][start] = TAG2ID['START']
			label2tag[label][end-1] = TAG2ID['END']

		# process data
		label2datatag = {}
		for name in LABEL2Q:
			label2datatag[name] = split_data(data, label2tag[name], len(LABEL2Q[name]))

		for label in label2datatag:
			total_list = label2datatag[label]
			for qdx, (data, tag) in enumerate(total_list):
				data = data.replace(' ', space).replace('　', space)
				data = list(data)

				# replace unk
				for idx, c in enumerate(data):
					if c not in tokenizer.vocab:
						data[idx] = '[UNK]'

				tup = list(LABEL2Q[label])+['[SEP]']+data
				forward_pad_tags = [TAG2ID['N']]*(len(LABEL2Q[label])+2)
				if len(tup)+1 != len(forward_pad_tags+tag):
					print(len(tup), len(forward_pad_tags+tag))
					return
				tags_to_be_padded.append(forward_pad_tags+tag)
				data_to_be_padded.append(tup)
				labels_len.append(len(LABEL2Q[label]))
				if qdx == 0:
					followed.append(0)
				else:
					followed.append(1)

	# merge tensors
	padded_data = tokenizer(data_to_be_padded, is_split_into_words=True, padding=True, return_tensors='pt')
	max_seq_len = padded_data['input_ids'].shape[1]
	for mdx, tags in enumerate(tags_to_be_padded):
		tags_to_be_padded[mdx].extend([TAG2ID['N']]*(max_seq_len-len(tags)))
	padded_tags = torch.tensor(tags_to_be_padded)
	followed = torch.tensor(followed)
	labels_len = torch.tensor(labels_len)
	return padded_data, padded_tags, followed, labels_len
	

def split_data(data, tags, length):
	MAX = 505 - length
	if MAX >= len(data):
		return [(data, tags)]

	data_list = []
	stop_list = [' ', '。', '　']
	start = 0
	tag = MAX
	i = tag

	# print(data)

	while True:
		if data[i] in stop_list:
			data_list.append((data[start:i], tags[start:i]))
			if i + MAX >= len(data):
				data_list.append((data[i:len(data)], tags[i:len(data)]))
				break
			tag = i + MAX
			start = i
			i = tag
		i -= 1

	return data_list

def divide_dataset(data, fold=1):
	np.random.seed(0)
	data_list = []
	for i in range(len(data)):
		data_list.append(data[i])
	data_list = np.array(data_list)
	np.random.shuffle(data_list)

	num_blocks = 10-fold
	start = 100*num_blocks
	a = np.vstack([data_list[0:start], data_list[start+100:]])

	# print('valid', start, start+50, 'test', start+50, start+100)
	return a, data_list[start:start+100]

def data2pixel(data):
	with open('data', 'wb') as f:
		pickle.dump(data, f)

def calculate_F1(logits, tags, labels_len):
	total_preds = []
	total_corrects = []

	for idx, batch_tags in enumerate(tags):
		prefix = labels_len[idx].item()
		c_start, p_start = -1, -1
		for i in range(prefix+2, batch_tags.shape[0]):
			correct, pred = batch_tags[i].item(), logits[idx][i].item()
			# total corrects
			if correct == 1:
				c_start = i
			elif c_start > 0 and correct == 2:
				total_corrects.append((c_start, i+1))
				c_start = -1
			# totoal preds
			if pred == 1:
				p_start = i
			elif p_start > 0 and pred == 2:
				total_preds.append((p_start, i+1))
				p_start = -1
	
	# zero situation
	if len(total_corrects) == 0 and len(total_preds) == 0:
		return 1
	elif len(total_corrects) == 0:
		return 0
	elif len(total_preds) == 0:
		return 0

	# normal situation
	# cauculate intersection
	pred_correct = 0
	for i in total_preds:
		if i in total_corrects:
			pred_correct += 1
	
	if pred_correct == 0:
		return 0
	
	precision = pred_correct / len(total_preds)
	recall = pred_correct / len(total_corrects)
	
	return 2*precision*recall/(precision+recall)

if __name__ == '__main__':
	a = [1,2]
	b = [3,4]
	print(a+b)