from __future__ import division
from dialog.TagSample.modelHelper import ShareEmbeddings, RNNEncoder
import tensorflow as tf

class TagSimple(tf.keras.Model):
    def __init__(self, shared_embeddings, rnn_encoder, output_size):
        super(TagSimple, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.rnn_encoder = rnn_encoder

        # self.classifier = tf.keras.Sequential(
        #     [tf.keras.layers.Dense(units=rnn_encoder.output_size, activation=tf.tanh),
        #      tf.keras.layers.Dense(units=rnn_encoder.output_size, activation=tf.tanh),
        #      tf.keras.layers.Dense(units=output_size),
        #      ]
        # )

        self.linear1 = tf.keras.layers.Dense(units=rnn_encoder.output_size, activation=tf.tanh)
        self.linear2 = tf.keras.layers.Dense(units=rnn_encoder.output_size, activation=tf.tanh)
        self.linear3 = tf.keras.layers.Dense(units=output_size)

    def call(self, inputs, training=None, mask=None):
        src_emb = self.shared_embeddings(inputs)
        encoder_outputs, (enc_hidden, _) = self.rnn_encoder(src_emb)
        logits = self.linear3(self.linear2(self.linear1(enc_hidden)))
        log_prob = tf.nn.log_softmax(logits, axis=-1)
        return log_prob

if __name__ == "__main__":
    # tf.enable_eager_execution()
    from dialog.TagSample.Dataset import get_dataset

    train_path = "/home/panxie/Documents/myGAN/multi-response/data/weibo/sampler_data.tsv"
    vocab_file = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/weibo.vocab.txt"
    ids_file = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/sampler_data_idx.csv"

    Dataset = get_dataset(vocab_file, train_path)

    Embedding = ShareEmbeddings(Dataset.vocab_size, 64)
    Encoder = RNNEncoder("gru", hidden_size=128, num_layers=1)
    model = TagSimple(Embedding, Encoder, output_size=Dataset.vocab_size)
    src, tgt = Dataset._train_input_fn(train_path, ids_file)
    log_prob = model(src)
    print(model.trainable_weights)

    print(log_prob.shape)
    
