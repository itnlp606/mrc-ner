import os
import copy
import torch
import shutil
import numpy as np
from time import time
from tqdm import tqdm
from models import BERTseq
from data_loader import load_data
from torch.optim import AdamW
from utils import divide_dataset, preprocessing, calculate_F1, split_data
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from constants import DEVICE_NAME, DEVICE, LABEL2Q

class Processor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, cache_dir='pretrained_models')

    def run(self):
        print('Running on', DEVICE_NAME)
        if self.args.is_train:
            if len(self.args.fold) > 1:
                for fold in range(self.args.fold[0], self.args.fold[1]+1):
                    train, valid = divide_dataset(load_data(), fold=fold)
                    self._train(train, valid, fold)
            else:
                train, valid = divide_dataset(load_data(), fold=self.args.fold[0])
                self._train(train, valid, fold=self.args.fold[0])
            
        else:
            self._predict()
    
    def _data2loader(self, data, mode):
        padded_data, padded_tags, followed, labels_len = preprocessing(data, self.tokenizer)
        data = TensorDataset(padded_data['input_ids'], padded_data['attention_mask'], padded_tags, labels_len)
        
        if mode == 'seq':
            sampler = RandomSampler(data)
        elif mode == 'rand':
            sampler = SequentialSampler(data)
        else:
            raise Exception("Wrong mode, please enter 'seq" or 'rand')

        loader = DataLoader(data, sampler=sampler, batch_size=self.args.batch_size)
        return loader, followed

    def _train(self, train, valid, fold):
        # init
        self.model = BERTseq(self.args).to(DEVICE)
        train_loader, _ = self._data2loader(train, mode='seq')
        valid_loader, _ = self._data2loader(valid, mode='rand')

        # optimizer and scheduler
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=1e-8)

        total_steps = 1000
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

        top, stop = 0, 0
        best_model = None
        best_epoch = None

        print('training start.. on fold', fold)
        for i in range(self.args.num_epoches):
            # training
            start_time = time()
            self.model.train()
            train_losses = 0
            for idx, batch_data in enumerate(train_loader):
                batch_data = tuple(i.to(DEVICE) for i in batch_data)
                ids, masks, tags, _ = batch_data

                self.model.zero_grad()
                loss, _ = self.model(ids, masks, tags)

                # process loss
                loss.backward()
                train_losses += loss.item()

                # tackle exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)

                optimizer.step()

            scheduler.step()
            train_losses /= len(train_loader)

            # evaluate
            self.model.eval()

            total_corrects, total_preds, pred_correct = 0, 0, 0
            with torch.no_grad():
                valid_losses = 0
                for idx, batch_data in enumerate(valid_loader):
                    batch_data = tuple(i.to(DEVICE) for i in batch_data)
                    ids, masks, tags, labels_len = batch_data
                    loss, logits = self.model(ids, masks, tags)
                    logits = torch.argmax(logits, dim=2)
                    tup = calculate_F1(logits, tags, labels_len)

                    # adding
                    total_corrects += tup[0]
                    total_preds += tup[1]
                    pred_correct += tup[2]

                    # process loss
                    valid_losses += loss.item()
                valid_losses /= len(valid_loader)

            # calculate F1
            precision = pred_correct / total_preds
            recall = pred_correct / total_corrects
            avg_F1 = 2*precision*recall/(precision+recall)

            if avg_F1 > top:
                best_model = copy.deepcopy(self.model)
                best_epoch = i+1
                top = avg_F1
                print('BREAK Epoch %d train_loss:%.2e valid_loss:%.2e precision:%.4f recall:%.4f F1:%.4f time %.0f' % \
                    (i+1, train_losses, valid_losses, precision, recall, avg_F1, time()-start_time))
                stop = 0
            else:
                if stop > self.args.stop_num:
                    torch.save(best_model, 'models/Mod' + str(fold) + '_' + str(best_epoch))
                    return
                stop += 1

                print('Epoch %d train_loss:%.2e valid_loss:%.2e precision:%.4f recall:%.4f F1:%.4f time:%.0f' % \
                    (i+1, train_losses, valid_losses, precision, recall, avg_F1, time()-start_time))
        

    def _predict(self):
        model_folder_name, data_folder_name, space = 'mrc-models', 'tcdata/juesai', ','
        modelList = os.listdir(model_folder_name)
        dataList = os.listdir(data_folder_name)            
        file2dic = {}

        # init ds, create empty file
        for data_file_name in dataList:
            prefix, _ = os.path.splitext(data_file_name)
            file2dic[prefix] = {}
            with open('result/result/'+prefix+'.ann', 'w', encoding='utf-8') as f:
                f.write('\n')

        for model in modelList:
            # load model, variables
            self.model = torch.load(model_folder_name + '/' + model, map_location=DEVICE)

            # process each file
            for file_dx, data_file_name in enumerate(tqdm(dataList)):
                # if file_dx == 3:
                #     break
                prefix, _ = os.path.splitext(data_file_name)
                with open(data_folder_name + '/' + data_file_name) as f:
                    content = f.read()

                # judge empty file
                if content == ' ':
                    continue

                content = content.replace(' ', space).replace('　', space)
                # for each label, get predictions
                for label in LABEL2Q:
                    # basic variable
                    label_len = len(LABEL2Q[label])
                    offset = 0

                    # split data and get logits
                    data_list = split_data(content, label_len)
                    for data in data_list:
                        # preprocessing data
                        data = list(data)
                        for idx, c in enumerate(data):
                            if c not in self.tokenizer.vocab:
                                data[idx] = '[UNK]'
                        qtr = list(LABEL2Q[label])+['[SEP]']+data

                        # feed into model
                        tokens = self.tokenizer(qtr, is_split_into_words=True, return_tensors='pt')
                        labels = torch.zeros(tokens['input_ids'].shape, dtype=torch.long).to(DEVICE)
                        _, logits = self.model(tokens['input_ids'].to(DEVICE), tokens['attention_mask'].to(DEVICE), labels=labels)
                        logits = logits.squeeze(0).argmax(1)

                        # extract entities
                        start, start_idx = 0, -1
                        for i in range(label_len+2, logits.shape[0]):
                            tag = logits[i].item()
                            if tag == 1:
                                start = 1
                                start_idx = i
                            elif tag == 2 and start & 1:
                                start_pos = start_idx - label_len - 2
                                end_pos = i - label_len - 2 + 1
                                entity = ''.join(data[start_pos: end_pos]) # caculated
                                real_entity = content[start_pos+offset: end_pos+offset] # real word in paragraph
                                if real_entity != entity and '[UNK]' not in entity:
                                    print('wrong match', real_entity, entity)
                                    return
                                
                                # put in dic
                                tup = (label, start_pos+offset, end_pos+offset, real_entity)
                                if tup in file2dic[prefix]:
                                    file2dic[prefix][tup] += 1
                                else:
                                    file2dic[prefix][tup] = 1

                                start = 0

                        offset += len(data)

        # write to files
        blade = 5
        for prefix in file2dic:
            print('write', prefix)
            with open('result/result/'+prefix+'.ann', 'w', encoding='utf-8') as f:
                preds = file2dic[prefix]
                for idx, tup in enumerate(preds):
                    if preds[tup] < blade:
                        continue
                    extra = '' if idx == 0 else '\n'
                    t = extra + 'T' + str(idx+1) + '\t' + tup[0] + ' ' + str(tup[1]) + ' ' + str(tup[2]) + '\t' + tup[3]
                    f.write(t)

        # make zip
        shutil.make_archive("result", 'zip', "result")

if __name__ == '__main__':
    a = [1, 2, 3]
    b = [4, 5, 67]
    print(a+b)