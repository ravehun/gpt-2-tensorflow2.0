import collections
import tensorflow as tf

PAD_ID = 0
UNKNOWN_ID = 1
START_ID = 3
END_ID = 4


def load_vocab(vocab_path):
    vocab = collections.OrderedDict()
    index = 0
    for line in open(vocab_path, 'r').read().splitlines():
        vocab[line.split()[0]] = index
        index += 1
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def parse_example(serialized_example):
    data_fields = {
        'targets': tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True, default_value=0.0),
        'open_f': tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True, default_value=0.0),
        'high_f': tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True, default_value=0.0),
        'low_f': tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True, default_value=0.0),
        'close_f': tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True, default_value=0.0),
        'volume_f': tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True, default_value=0.0),
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)

    targets = tf.cast(parsed['targets'], tf.float32)
    open_f = tf.cast(parsed['open_f'], tf.float32)
    high_f = tf.cast(parsed['high_f'], tf.float32)
    low_f = tf.cast(parsed['low_f'], tf.float32)
    close_f = tf.cast(parsed['close_f'], tf.float32)
    volume_f = tf.cast(parsed['volume_f'], tf.float32)

    inputs = tf.stack(
        [
            open_f,
            high_f,
            low_f,
            close_f,
            volume_f,
        ],
        axis=-1
    )
    targets = tf.expand_dims(targets,-1)
    return inputs, targets


def input_fn(tf_records, batch_size=32, padded_shapes=([-1,-1,], [-1,-1,]), epoch=10, buffer_size=10000):
    if type(tf_records) is str:
        tf_records = [tf_records]
    dataset = tf.data.TFRecordDataset(tf_records, buffer_size=10000)
    dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.map(parse_example)
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
