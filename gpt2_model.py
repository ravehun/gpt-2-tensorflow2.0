from layers.feed_forward import *
from layers.attention_layer import *
from layers.embedding_layer import *
from layers.layer_norm import LayerNormalization
from tensorflow.python.framework import tensor_shape
from utils.tf_utils import *
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"

train_step_signature = [
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32, name="Inputs"),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32, name="Targets"),
    tf.TensorSpec(shape=(None), dtype=tf.int32, name="Step")
]


class Gpt2(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, max_seq_len, dense_feature_dim,
                 optimizer="adam", learning_rate=1e-3, rev_embedding_projection=False):
        super(Gpt2, self).__init__()

        self.rev_embedding_projection = rev_embedding_projection
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.max_seq_len = max_seq_len
        self.dense_feature_dim = dense_feature_dim
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.optimizer_t = optimizer
        self.dataset = None
        self.mirrored_strategy = None

        self.input_projection = Conv1d(5, d_model)

        self.pos_embedding = PositionEmbeddingLayer(
            self.max_seq_len, self.d_model)

        self.decoder_layers = [DecoderLayer(self.d_model, self.num_heads, self.dff)
                               for _ in range(self.num_layers)]
        self.layer_norm = LayerNormalization(self.d_model)

        if not self.rev_embedding_projection:
            self.output_layer = OutputLayer(1)

        self.loss_object = tf.keras.losses.MeanSquaredLogarithmicError(reduction='none')

        self.accuracy_object = tf.keras.metrics.MeanAbsolutePercentageError(
            name='accuracy')

        self.train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

    def call(self, x, training=True, past=None):
        # x = tf.cast(x, tf.int32)
        batch, sequence, dim = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        if past is None:
            pasts = [None] * self.num_layers
        else:
            pasts = past

        assert len(pasts) == self.num_layers

        att_mask = create_masks(x[..., 1])
        past_length = 1 if past is None else tf.shape(past)[-2]


        with tf.name_scope("embeddings"):
            embedded_x = self.input_projection(x)
            hidden_states = embedded_x + self.pos_embedding(x[..., -1], start=past_length)

        presents = []
        for decoder_layer, past in zip(self.decoder_layers, pasts):
            hidden_states, present = decoder_layer(hidden_states, training, att_mask, past=past)
            presents.append(present)

        hidden_states = self.layer_norm(hidden_states)

        if self.rev_embedding_projection:
            logits = self.input_projection(hidden_states, mode="projection")
        else:
            logits = self.output_layer(hidden_states)

        return logits, presents

    def get_padded_metric(self, labels, logits, eps=1e-9):
        with tf.name_scope("padded_accuracy"):
            labels = tf.squeeze(labels, -1)
            weights = tf.cast(tf.not_equal(labels, 0), tf.float32)

            nonpad_seq = tf.reduce_sum(weights, axis=1,keepdims=True) + eps
            acc = tf.cast(self.accuracy_object(logits, labels), tf.float32)

            metric = tf.reduce_sum(tf.cast(acc * weights, tf.float32), axis=1, keepdims=True) / nonpad_seq

            metric = tf.reduce_mean(metric)
            return tf.cast(metric, tf.float32)

    def creat_optimizer(self):
        optimizer = self.optimizer_t.lower()
        with tf.name_scope("optimizer"):
            if optimizer == "adam":
                self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
                                                          epsilon=1e-9)
            elif optimizer == "adadelxxta":
                self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
            elif optimizer == "rms":
                self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            else:
                self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
            return self.optimizer

    def get_loss(self, real, pred):
        with tf.name_scope("loss_layer"):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = self.loss_object(real, pred)
            with tf.name_scope("loss_masking"):
                mask = tf.cast(mask, dtype=loss_.dtype)
                mask = tf.squeeze(mask, -1)
                loss_ *= mask
            loss_ = tf.reduce_sum(loss_, axis=1)
            sequence_avg_loss = loss_ / tf.reduce_sum(mask, axis=1)
            return sequence_avg_loss

    def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
        with tf.name_scope('checkpoint_manager'):
            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

            if load_model:  # If want to load trained weights
                ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored...............')
            else:
                print("Initializing model from scratch..........")

    def load_model(self, filepath):
        ckpt = tf.train.Checkpoint(model=self)
        ckpt_manager = tf.train.CheckpointManager(ckpt, filepath)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Model Restored..........................")

    def create_summary_writer(self, summary_path):
        train_summary_path = summary_path + "/train"
        test_summary_path = summary_path + "/test"

        with tf.name_scope('summary'):
            self.train_writer = tf.summary.create_file_writer(train_summary_path)
            self.test_writer = tf.summary.create_file_writer(test_summary_path)

            return self.train_writer, self.test_writer

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inputs, targets, step, grad_clip=True, clip_value=2.5):

        with tf.GradientTape() as tape:
            predictions, _ = self(inputs, training=True)
            # print(targets.shape, predictions.shape)
            loss = tf.reduce_mean(self.get_loss(targets, predictions))
        with tf.name_scope("gradients"):
            gradients = tape.gradient(loss, self.trainable_variables)
            if grad_clip:
                gradients = [(tf.clip_by_value(grad, -clip_value, clip_value))
                             for grad in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


        mean_abs_error = self.get_padded_metric(targets, predictions)

        with tf.name_scope("summary_writer"):
            with self.train_writer.as_default():
                tf.summary.scalar("loss", loss, step=tf.cast(step, tf.int64))
                tf.summary.scalar("mean_abs_error", mean_abs_error, step=tf.cast(step, tf.int64))

        return loss, mean_abs_error

    @tf.function
    def distributed_train_step(self, inputs, targets, step, grad_clip=True, clip_value=1.0):
        def step_fn(inp, tar):
            with tf.GradientTape() as tape:
                logits = self(inputs)
                cross_entropy = self.get_loss(targets, logits)
                loss = tf.reduce_mean(cross_entropy)

            with tf.name_scope("gradients"):
                gradients = tape.gradient(loss, self.trainable_variables)
                if grad_clip:
                    gradients = [(tf.clip_by_value(grad, -clip_value, clip_value))
                                 for grad in gradients]
                self.optimizer.apply_gradients(list(zip(gradients, self.trainable_variables)))
            return cross_entropy

        per_example_losses = self.mirrored_strategy.experimental_run_v2(
            step_fn, args=(inputs, targets))
        mean_loss = self.mirrored_strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

        with tf.name_scope("summary_writer"):
            with self.train_writer.as_default():
                tf.summary.scalar("loss", mean_loss, step=step)
        return mean_loss

    def fit(self, train_dataset):
        if self.mirrored_strategy is None:
            tf.summary.trace_on(graph=True, profiler=True)
            for (step, (inputs, targets)) in enumerate(train_dataset):
                train_loss, train_acc = self.train_step(inputs, targets, step)
                if step % 10 == 0:
                    print('Step {} Train_Loss {:.4f} Train_Accuracy {:.4f}'.format(
                        step, train_loss, train_acc))

                if step == 0:
                    with self.train_writer.as_default():
                        tf.summary.trace_export(
                            name="gpt-2",
                            step=0,
                            profiler_outdir=LOG_DIR)

                if step % 1000 == 0:
                    ckpt_save_path = self.ckpt_manager.save()
                    print('Saving checkpoint for step {} at {}'.format(step,
                                                                       ckpt_save_path))
        else:
            with self.mirrored_strategy.scope():
                tf.summary.trace_on(graph=True, profiler=True)
                for (step, (inputs)) in enumerate(train_dataset):
                    train_loss = self.distributed_train_step(inputs, step)
                    if step == 0:
                        with self.train_writer.as_default():
                            tf.summary.trace_export(
                                name="gpt-2",
                                step=0,
                                profiler_outdir=LOG_DIR)
                    if step % 100 == 0:
                        print('Step {} Train_Loss {:.4f}'.format(
                            step, train_loss))
                    if step % 1000 == 0:
                        ckpt_save_path = self.ckpt_manager.save()
                        print('Saving checkpoint for step {} at {}'.format(step,
                                                                           ckpt_save_path))


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, proj_weights=None, kernel_initializer=None):
        super(OutputLayer, self).__init__()
        self.proj_weights = proj_weights
        self.output_dim = output_dim
        self.layer_weights = None
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        if self.proj_weights is None:
            input_dim = tensor_shape.dimension_value(input_shape[-1])
            self.layer_weights = self.add_weight(
                'output_layer_weights',
                shape=[input_dim, self.output_dim],
                initializer=self.kernel_initializer,
                trainable=True)
        super(OutputLayer, self).build(input_shape)

    def call(self, x):
        batch, sequence, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1]
        h_flat = tf.reshape(x, [-1, d_model])

        if self.proj_weights is None:
            out = tf.matmul(h_flat, self.layer_weights)
        else:
            out = tf.matmul(h_flat, self.porj_weights, transpose_b=True)
        out = tf.reshape(out, [batch, sequence, self.output_dim])
        return out


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,
                 dr_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dr_rate = dr_rate

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.feed_forward = FeedForward(self.d_model, self.dff, self.dr_rate)
        self.layer_norm1 = LayerNormalization(self.d_model)
        self.layer_norm2 = LayerNormalization(self.d_model)

    def call(self, x, training, mask, past=None):
        out, present = self.mha(self.layer_norm1(x), mask=mask, past_layer=past,
                                training=training)  # (batch_size, input_seq_len, d_model)
        with tf.name_scope("residual_conn"):
            x = x + out
        out = self.feed_forward(self.layer_norm2(x), training=training)  # (batch_size, input_seq_len, d_model)
        with tf.name_scope("residual_conn"):
            x = x + out
        return x, present
