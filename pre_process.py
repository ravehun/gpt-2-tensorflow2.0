import csv
import datetime
import glob
import os
from collections import Counter

import click
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import tqdm
from ftfy import fix_text
import traceback
import pandas as pd

_ROOT = os.path.abspath(os.path.dirname(__file__))
PROCESS_DATA_PATH = _ROOT + "/data/processed.txt"
BPE_TSV_PATH = _ROOT + "/data/bpe_spm.tsv"
BPE_MODEL_PATH = _ROOT + "/data/bpe_model"
TF_RECORDS = _ROOT + "/data/stock_tf_records/"
BOS_ID = 3
EOS_ID = 4


def process_text(text_files):
    print("Pre-processing the text data.....")
    file_writer = open(PROCESS_DATA_PATH, "w")
    for file_name in tqdm.tqdm(text_files):
        fr = open(file_name, 'r')
        file_writer.writelines([fix_text(line, normalization='NFKC') for line in fr.readlines()])
        fr.close
    file_writer.close()


def train_byte_pair_encoding(vocab_size):
    print("Training BytePair encoding......")
    token_dict = Counter()
    with open(PROCESS_DATA_PATH, 'r') as fr:
        for line in tqdm.tqdm(fr):
            token_dict.update(line.lower().split())

    with open(BPE_TSV_PATH, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for word in token_dict:
            tsv_output.writerow([word, token_dict[word]])

    spmcmd = '--input={spm_input} --model_prefix={spm_model} --input_format=tsv --vocab_size={vocab_size} --user_defined_symbols=[SEP],[BOS],[EOS] --hard_vocab_limit=false --model_type=bpe --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK]'.format(
        spm_input=BPE_TSV_PATH, spm_model=BPE_MODEL_PATH, vocab_size=vocab_size)
    spm.SentencePieceTrainer.train(spmcmd)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _string_feature(value):
    return tf.train.Feature(byte_list=tf.train.BytesList(value=value))


def get_stock_name(x):
    return x.split("/")[-1][:-4]


def serialize_example(targets, open_f, high_f, low_f, close_f, volume_f):
    feature = {
        'targets': _float_feature(targets),
        'open_f': _float_feature(open_f),
        'high_f': _float_feature(high_f),
        'low_f': _float_feature(low_f),
        'close_f': _float_feature(close_f),
        'volume_f': _float_feature(volume_f),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tf_records(text_files, min_seq_len, max_seq_len, per_file_limit=50000
                      , train_date='2008-01-01'
                      , valid_date='2014-01-01'
                      ):
    print("Creating TF Records...............")
    if not os.path.exists(TF_RECORDS):
        os.makedirs(TF_RECORDS)
    filename = TF_RECORDS + str(datetime.datetime.now().timestamp()) + ".tfrecord"
    tf_writer = tf.io.TFRecordWriter(filename)
    doc_counts = 0
    for filename in tqdm.tqdm(text_files):
        try:
            df = pd.read_csv(filename)
            stock_name = get_stock_name(filename)
            feature = [
                # 'Date',
                'Open',
                'High',
                'Low',
                'Close',
                'Volume',
                # 'OpenInt'
            ]
            df = df[(df.Date >= train_date) & (df.Date < valid_date)]
            x = df[feature].values
            if max_seq_len > x.shape[0] > min_seq_len:
                inputs = np.concatenate([x, np.zeros([1, len(feature)])], axis=0).astype(np.float32)
                open_f = inputs[:, 0].tolist()
                high_f = inputs[:, 1].tolist()
                low_f = inputs[:, 2].tolist()
                close_f = inputs[:, 3].tolist()
                volume_f = inputs[:, 4].tolist()
                targets = np.concatenate([np.zeros(1), x[:, 3]], axis=0).astype(np.float32).tolist()

                example = serialize_example(targets, open_f, high_f, low_f, close_f, volume_f)
                tf_writer.write(example)
        except:
            traceback.print_exc()

    tf_writer.close()


@click.command()
@click.option('--data-dir', type=str, default="./data/Stocks", show_default=True, help="training data path")
@click.option('--min-seq-len', type=int, default=800, show_default=True, help="minimum sequence length")
@click.option('--max-seq-len', type=int, default=1500, show_default=True, help="minimum sequence length")
@click.option('--train-date', type=str, default='2009-01-01', show_default=True, help="example start")
@click.option('--valid-date', type=str, default='2014-01-01', show_default=True, help="example end")
def train(data_dir, min_seq_len, max_seq_len,train_date, valid_date):
    text_files = glob.glob((data_dir + "/*"))
    create_tf_records(text_files, min_seq_len, max_seq_len, train_date=train_date, valid_date=valid_date)
    print("Pre-processing is done............")


if __name__ == "__main__":
    train()
