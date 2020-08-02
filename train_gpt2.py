import click
import glob
from gpt2_model import *
from data_pipeline import input_fn
import tensorflow as tf
import os
import json

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"


@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=1501, show_default=True, help="Seq length")
@click.option('--dense-feature-dim', type=int, default=5, show_default=True, help="Vocab size")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=4, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--distributed', type=bool, default=False, show_default=True, help="distributed training")
def train(num_layers, embedding_size, num_heads, dff, max_seq_len, dense_feature_dim,
          optimizer="adam", batch_size=16, learning_rate=1e-3, distributed=False):
    
    
    par_map = {"num_layers": num_layers, "d_model": embedding_size,
               "num_heads": num_heads, "dff": dff,
               "max_seq_len": max_seq_len, "vocab_size": dense_feature_dim}

    exp_name = "_".join(['{}_{}'.format(k, v) for k, v in par_map.items()])

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(MODEL_DIR + '/model_par.json', 'w') as f:
        json.dump(par_map, f)
        
        
        
    tf_records = glob.glob((_ROOT + "/data/stock_tf_records/*.tfrecord"))
    if distributed:
        dist_dataset = input_fn(tf_records, batch_size=batch_size)
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dist_dataset)
        with mirrored_strategy.scope():

            model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, dense_feature_dim,
                         optimizer=optimizer, learning_rate=learning_rate)
            model.creat_optimizer()
            model.create_checkpoint_manager(MODEL_DIR)
            model.create_summary_writer(LOG_DIR)

        model.mirrored_strategy = mirrored_strategy
        model.fit(dist_dataset)
    else:
        dataset = input_fn(tf_records, batch_size=batch_size)
        model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, dense_feature_dim,
                     optimizer=optimizer, learning_rate=learning_rate)
        model.creat_optimizer()
        model.create_checkpoint_manager(MODEL_DIR)
        model.create_summary_writer(LOG_DIR)
        model.fit(dataset)
        print("Training Done................")


if __name__ == "__main__":
    train()
