from __future__ import division
import tensorflow as tf


class TagSimple(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size, rnn_type, bidirectional=True):
        super(TagSimple, self).__init__()
        # embedding layer
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        if self.rnn_type == "gru":
            self.rnn_encoder = tf.keras.layers.GRU(units=hidden_size,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   go_backwards=bidirectional)
        elif self.rnn_type == "lstm":
            self.rnn_encoder = tf.keras.layers.LSTM(units=hidden_size,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   go_backwards=bidirectional)
        else:
            raise NameError("rnn typr must be lstm or gru")

        self.linear1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.tanh)
        self.linear2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.tanh)
        self.linear3 = tf.keras.layers.Dense(units=vocab_size)

    def call(self, inputs, training=None, mask=None):
        src_emb = self.embedding(inputs)
        if self.rnn_type == "gru":
            encoder_outputs, enc_hidden = self.rnn_encoder(src_emb)
        else:
            encoder_outputs, enc_hidden, _ = self.rnn_encoder(src_emb)
        logits = self.linear3(self.linear2(self.linear1(enc_hidden)))
        log_prob = tf.nn.log_softmax(logits, axis=-1)
        return log_prob


if __name__ == "__main__":
    # tf.enable_eager_execution()
    #from dialog.TagSample.Dataset import get_dataset

    train_path = "/home/panxie/Documents/myGAN/multi-response/data/weibo/sampler_data.tsv"
    vocab_file = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/weibo.vocab.txt"
    ids_file = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/sampler_data_idx.csv"

        
    model = TagSimple(100, 64, 100, "lstm")
    src = tf.constant([[2,3,4], [5,6,7]], dtype=tf.int32)
    log_prob = model(src)
    for var in model.trainable_weights:
        print(var)
    print(log_prob.shape)


