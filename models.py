from torch import nn
from constants import ID2TAG
from transformers import AutoModelForTokenClassification, AutoConfig

class BERTseq(nn.Module):
    def __init__(self, args):
        super(BERTseq, self).__init__()
        self.emission = AutoModelForTokenClassification.from_pretrained(args.pretrained_model, cache_dir='pretrained_models', num_labels=len(ID2TAG))

    def forward(self, ids, masks, labels):
        return self.emission(input_ids=ids, attention_mask=masks, labels=labels)

class Matchnn(nn.Module):
    def __init__(self):
        super(Matchnn, self).__init__()