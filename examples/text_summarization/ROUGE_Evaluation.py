import os
import shutil
import sys
from tempfile import TemporaryDirectory
import torch

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.dataset.swiss import SwissSummarizationDataset
from utils_nlp.dataset.bundes import BundesSummarizationDataset

from utils_nlp.eval import compute_rouge_python, compute_rouge_perl
from utils_nlp.models.transformers.extractive_summarization import (
    ExtractiveSummarizer,
    ExtSumProcessedData,
    ExtSumProcessor,
)

from utils_nlp.models.transformers.datasets import SummarizationDataset
import nltk
from nltk import tokenize

import pandas as pd
import scrapbook as sb
import pprint


CACHE_DIR = TemporaryDirectory().name

BUNDES_DATA_PATH='/home/ubuntu/data/bundes_dataset/'
SWISS_DATA_PATH='/home/ubuntu/data/swiss_dataset/'

bundes_save_path = os.path.join(BUNDES_DATA_PATH)
# bundes_train = torch.load(os.path.join(bundes_save_path, "train_full202008111812.pt"))
bundes_test = torch.load(os.path.join(bundes_save_path, "test_full202008111812.pt"))


swiss_save_path = os.path.join(SWISS_DATA_PATH)
# swiss_train = torch.load(os.path.join(swiss_save_path, "train_full.pt"))
swiss_test = torch.load(os.path.join(swiss_save_path, "test_full.pt"))

# models:


model_names = ['distilbert-base-german-cased', 'bert-base-german-cased']
train_names = ['200805_distilbert-base-german-cased_swiss', 
               '200806_bert-base-german-cased_swiss',
               '200811_bert-base-german-cased_swissBundes',
               '200811_distilbert-base-german-cased_swissBundes',
               '200812_bert-base-german-cased_bundes',
               '200812_distilbert-base-german-cased_bundes',
               'lead_1',
               'lead_2',
               'lead_3',
              ]

model_filepaths = ['/home/ubuntu/models/200805_distilbert-base-german-cased_swiss/output/',
                   '/home/ubuntu/models/200806_bert-base-german-cased_swiss/output/',
                   '/home/ubuntu/models/200811_bert-base-german-cased_swissBundes/output/',
                   '/home/ubuntu/models/200811_distilbert-base-german-cased_swissBundes/output/',
                   '/home/ubuntu/models/200812_bert-base-german-cased_bundes/output/',
                   '/home/ubuntu/models/200812_distilbert-base-german-cased_bundes/output/',
                  ]

models = {
    '200805_distilbert-base-german-cased_swiss':{
        'model': 'distilbert-base-german-cased',
        'filepath': '/home/ubuntu/models/200805_distilbert-base-german-cased_swiss/output/'
    },
    '200806_bert-base-german-cased_swiss':{
        'model': 'bert-base-german-cased',
        'filepath': '/home/ubuntu/models/200806_bert-base-german-cased_swiss/output/'
    },
    '200811_bert-base-german-cased_swissBundes':{
        'model': 'bert-base-german-cased',
        'filepath': '/home/ubuntu/models/200811_bert-base-german-cased_swissBundes/output/'
    },
    '200811_distilbert-base-german-cased_swissBundes':{
        'model': 'distilbert-base-german-cased',
        'filepath': '/home/ubuntu/models/200811_distilbert-base-german-cased_swissBundes/output/'
    },
    '200812_bert-base-german-cased_bundes':{
        'model': 'bert-base-german-cased',
        'filepath': '/home/ubuntu/models/200812_bert-base-german-cased_bundes/output/'
    },
    '200812_distilbert-base-german-cased_bundes':{
        'model': 'distilbert-base-german-cased',
        'filepath': '/home/ubuntu/models/200812_distilbert-base-german-cased_bundes/output/'
    },
    
}

MAX_POS_LENGTH = 512


# GPU used for training
NUM_GPUS = torch.cuda.device_count()

# Encoder name. Options are: 1. baseline, classifier, transformer, rnn.
ENCODER = "transformer"

# How often the statistics reports show up in training, unit is step.
REPORT_EVERY=50


# create processors:
processors = {}
for model_name in model_names:
    processors[model_name] = ExtSumProcessor(model_name=model_name, cache_dir=CACHE_DIR)
    

summarizers= {}
model_filename = "dist_extsum_model.pt"

for model, meta in list(models.items()):
    print("creating summarizer for", model)

    processor = processors[meta['model']]
    print("Processor loaded for", meta['model'])
    
    model_path = os.path.join(meta['filepath'], model_filename)
    summarizer = ExtractiveSummarizer(processor, meta['model'], ENCODER, MAX_POS_LENGTH, CACHE_DIR)
    summarizer.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("model loaded for", meta['model'])
    summarizers[model] = summarizer

source = {}
target = {}


source['bundes'] = []
source['swiss'] = []

temp_target_bundes = []
temp_target_swiss = []
for i in bundes_test:
    source['bundes'].append(i["src_txt"]) 
    
    temp_target_bundes.append(" ".join(j) for j in i['tgt']) 
target['bundes'] = [''.join(i) for i in list(temp_target_bundes)]

for i in swiss_test:
    source['swiss'].append(i["src_txt"]) 
    
    temp_target_swiss.append(" ".join(j) for j in i['tgt']) 
target['swiss'] = [''.join(i) for i in list(temp_target_swiss)]


### create test dictionary
torch_tests = {
    'bundes': bundes_test,
    'swiss': swiss_test
}


%%time
sentence_separator = "\n"
batch_size = 250
rouge_scores = {}
predictions = {}

TEST = False


for dataset in ['swiss', 'bundes']:
    predictions[dataset] = {}
    rouge_scores[dataset] = {}
    print("Dataset: ", dataset)
    if TEST:
        n = 5
    else:
        n = len(torch_tests[dataset])
    print("Sample size:", n)
    
    for train_name, summarizer in summarizers.items():
        print("model name: ", train_name)
        if "lead" in train_name:
            print("IN HERE")
            predictions[dataset][train_name] = leads[train_name][:n]
        else:
            predictions[dataset][train_name] = summarizer.predict(torch_tests[dataset][:n], num_gpus=0, batch_size=batch_size, sentence_separator=sentence_separator)
        
        rouge_scores[dataset][train_name] = compute_rouge_python(cand=predictions[dataset][train_name], ref=target[dataset][:n])

        
# print out the calculated rouge scores
pprint.pprint(rouge_scores)
