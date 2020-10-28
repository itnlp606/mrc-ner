import copy
import torch
from time import time
from models import BERTseq
from data_loader import load_data
from torch.optim import AdamW
from utils import divide_dataset, preprocessing
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from constants import DEVICE_NAME, DEVICE

class Processor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, cache_dir='pretrained_models')

    def run(self):
        print('Running on', DEVICE_NAME)
        if self.args.is_train:
            self.model = BERTseq(self.args).to(DEVICE)
            if len(self.args.fold) > 1:
                for fold in range(self.args.fold[0], self.args.fold[1]+1):
                    train, valid = divide_dataset(load_data(), fold=fold)
                    self._train(train, valid)
            else:
                train, valid = divide_dataset(load_data(), fold=self.args.fold[0])
                self._train(train, valid, fold=self.args.fold[0])
            
        else:
            self._predict()
    
    def _data2loader(self, data, mode):
        padded_data, padded_tags, followed = preprocessing(data, self.tokenizer)
        data = TensorDataset(padded_data['input_ids'], padded_data['attention_mask'], padded_tags)
        
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

        top, stop = 500, 0
        best_model = None
        start_time = time()
        for i in range(self.args.num_epoches):
            # training
            self.model.train()
            train_losses = 0
            for idx, batch_data in enumerate(train_loader):
                batch_data = tuple(i.to(DEVICE) for i in batch_data)
                ids, masks, labels = batch_data

                self.model.zero_grad()
                loss, _ = self.model(ids, masks, labels)

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

            with torch.no_grad():
                valid_losses = 0
                for idx, batch_data in enumerate(valid_loader):
                    batch_data = tuple(i.to(DEVICE) for i in batch_data)
                    ids, masks, labels = batch_data
                    loss, _ = self.model(ids, masks, labels)

                    # process loss
                    valid_losses += loss.item()
                valid_losses /= len(valid_loader)

            if valid_losses < top:
                best_model = copy.deepcopy(self.model)
                top = valid_losses
                print('save new top', top)
                stop = 0
            else:
                if stop > self.args.stop_num:
                    torch.save(best_model, 'models/Mod' + str(fold) + '_' + str(i+1))
                    return
                stop += 1

            print('Epoch', i, train_losses, valid_losses, time()-start_time)
            start_time = time()
        

    def _predict(self):
        pass