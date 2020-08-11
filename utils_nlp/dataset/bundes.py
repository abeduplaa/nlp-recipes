# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/BertSum

"""
    Utility functions for downloading, extracting, and reading the
    Swiss dataset at https://drive.switch.ch/index.php/s/YoyW9S8yml7wVhN.
"""

import nltk

# nltk.download("punkt")
from nltk import tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os
import regex as re
# from torchtext.utils import extract_archive
import pandas
from sklearn.model_selection import train_test_split

from utils_nlp.dataset.url_utils import (
    maybe_download,
    maybe_download_googledrive,
    extract_zip,
)
from utils_nlp.models.transformers.datasets import (
    SummarizationDataset,
    IterableSummarizationDataset,
)



def _clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", lambda m: REMAP.get(m.group()), x,
    )


def _remove_ttags(line):
    line = re.sub(r"<t>", "", line)
    # change </t> to <q>
    # pyrouge test requires <q> as sentence splitter
    line = re.sub(r"</t>", "<q>", line)
    return line



def _target_sentence_tokenization(line):
    return line.split("<q>")


def join(sentences):
    return " ".join(sentences)


def BundesSummarizationDataset(top_n=-1, validation=False, prepare_extractive=True, language='german',CSV_PATH=None):
    """Load the bundes dataset by faktual."""
    
    if CSV_PATH is None:
        CSV_PATH ='/home/ubuntu/mnt/data/bundes_dataset/csv/bundes_data.csv'
        # FILE_NAME = "bundes_data.csv"
        
    train = pandas.read_csv(CSV_PATH).values.tolist()
    if(top_n!=-1):
        train = train[0:top_n]
    source = [str(item[0]) for item in train]
    summary = [str(item[1]) for item in train] 
    
    print("source[0]: ", source[0])
    print("summary[0]: ", summary[0])
    train_source,test_source,train_summary,test_summary=train_test_split(
        source,
        summary,
        train_size=0.95,
        test_size=0.05,
        random_state=123
    )
    
    if prepare_extractive:
        if validation:
            train_source, validation_source, train_summary, validation_summary = train_test_split(
                train_source, train_summary, train_size=0.9, test_size=0.1, random_state=123
            )
            return (
                SummarizationDataset(
                    source_file=None,
                    source=train_source,
                    target=train_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[                    
                        _clean,
                        _remove_ttags,
                        _target_sentence_tokenization,],
                    word_tokenize=nltk.word_tokenize,
                    top_n=top_n,
                    language=language,
                ),
                SummarizationDataset(
                    source_file=None,
                    source=validation_source,
                    target=validation_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[_clean,
                        _remove_ttags,
                        _target_sentence_tokenization,],
                    word_tokenize=nltk.word_tokenize,
                    top_n=top_n,
                    language=language,
                ),
                SummarizationDataset(
                    source_file=None,
                    source=test_source,
                    target=test_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[_clean,
                        _remove_ttags,
                        _target_sentence_tokenization,],
                    word_tokenize=nltk.word_tokenize,
                    top_n=top_n,
                    language=language,
                ),
            )
        else:
            return (
                SummarizationDataset(
                    source_file=None,
                    source=train_source,
                    target=train_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[_clean,
                        _remove_ttags,
                        _target_sentence_tokenization,],
                    word_tokenize=nltk.word_tokenize,
                    top_n=top_n,
                    language=language,
                ),
                SummarizationDataset(
                    source_file=None,
                    source=test_source,
                    target=test_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[_clean,
                        _remove_ttags,
                        _target_sentence_tokenization,],
                    word_tokenize=nltk.word_tokenize,
                    top_n=top_n,
                    language=language,
                ),
            )
    else:
        if validation:
            train_source, validation_source, train_summary, validation_summary = train_test_split(
                train_source, train_summary, train_size=0.9, test_size=0.1, random_state=123
            )
            return (
                SummarizationDataset(
                    source_file=None,
                    source=train_source,
                    target=train_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[
                        tokenize.sent_tokenize,
                    ],
                    top_n=top_n,
                ),
                SummarizationDataset(
                    source_file=None,
                    source=validation_source,
                    target=validation_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[
                        tokenize.sent_tokenize,
                    ],
                    top_n=top_n,
                ),
                SummarizationDataset(
                    source_file=None,
                    source=test_source,
                    target=test_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[
                        tokenize.sent_tokenize,
                    ],
                    top_n=top_n,
                ),
            )
        else:
            return (
                SummarizationDataset(
                    source_file=None,
                    source=train_source,
                    target=train_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[
                        tokenize.sent_tokenize,
                    ],
                    top_n=top_n,
                ),
                SummarizationDataset(
                    source_file=None,
                    source=test_source,
                    target=test_summary,
                    source_preprocessing=[tokenize.sent_tokenize],
                    target_preprocessing=[
                        tokenize.sent_tokenize,
                    ],
                    top_n=top_n,
                ),
            )