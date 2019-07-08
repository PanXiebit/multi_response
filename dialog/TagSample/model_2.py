# -*- encoding=utf8 -*-
from __future__ import division
# from dialog.TagSample.modelHelper import ShareEmbeddings, RNNEncoder
import tensorflow as tf


class TagSimple(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size, rnn_type,
                 num_layers=1, dropout=0.1, bidirectional=False):
        super(TagSimple, self).__init__()
        # embedding layer
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        # self.rnn_encoder = tf.keras.layers.GRU(units=hidden_size,
        #                                        return_sequences=True,
        #                                        return_state=True,
        #                                        go_backwards=False)

        if self.rnn_type == "lstm":
            self.rnn_cells = tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.LSTMCell(units=self.hidden_size,
                                          dropout=self.dropout)
                 for _ in range(self.num_layers)])

        elif self.rnn_type == "gru":
            self.rnn_cells = tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.GRUCell(units=self.hidden_size,
                                         dropout=self.dropout)
                 for _ in range(self.num_layers)])
        else:
            raise NameError("rnn_type must be 'lstm' or 'gru'.")

        self.rnn_encoder = tf.keras.layers.RNN(
            cell=self.rnn_cells,
            return_sequences=True,
            return_state=True,
            go_backwards=self.bidirectional
        )

        self.linear1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.tanh)
        self.linear2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.tanh)
        self.linear3 = tf.keras.layers.Dense(units=vocab_size)

    def call(self, inputs, training=None, mask=None):
        src_emb = self.embedding(inputs)
        if self.rnn_type == "lstm":
            _, (enc_hidden, _) = self.rnn_encoder(src_emb)
        elif self.rnn_type == "gru":
            outputs = self.rnn_encoder(src_emb)
            enc_hidden = tf.concat(outputs[1:], axis=-1)
        else:
            raise NameError
        logits = self.linear3(self.linear2(self.linear1(enc_hidden)))
        log_prob = tf.nn.log_softmax(logits, axis=-1)
        return log_prob
if __name__ == "__main__":
    #tf.enable_eager_execution()
    from dialog.TagSample.Dataset import get_dataset

    train_path = "/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/sampler_data.tsv"
    vocab_file = "/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/weibo.vocab.txt"
    ids_file = "/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/sampler_data_idx.csv"

    Dataset = get_dataset(vocab_file, train_path, vocab_size=50000)
    model = TagSimple(Dataset.vocab_size, embedding_size=64, hidden_size=100, rnn_type="gru", num_layers=2)
    src, tgt = Dataset._train_input_fn(train_path, ids_file)
    log_prob = model(src)
    for var in model.trainable_weights:
        print(var)
    # print(model.trainable_weights)

    print(log_prob.shape)
