import tensorflow as tf

class ShareEmbeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size):
        super(ShareEmbeddings, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)

    def call(self, inputs, **kwargs):
        embedded = self.embedding(inputs)
        return embedded

class RNNEncoder(tf.keras.layers.Layer):
    def __init__(self, rnn_type, hidden_size,
                 num_layers=1, dropout=0.1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        self.output_size = hidden_size
        hidden_size = hidden_size // num_directions
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # if self.rnn_type == "lstm":
        #     self.rnn_cells = tf.keras.layers.StackedRNNCells(
        #         [tf.keras.layers.LSTMCell(units=self.output_size,
        #                                   dropout=self.dropout)
        #          for _ in range(self.num_layers)])
        # 
        # elif self.rnn_type == "gru":
        #     self.rnn_cells = tf.keras.layers.StackedRNNCells(
        #         [tf.keras.layers.GRUCell(units=self.output_size,
        #                                  dropout=self.dropout)
        #          for _ in range(self.num_layers)])
        # 
        # self.rnn = tf.keras.layers.RNN(
        #     cell=self.rnn_cells,
        #     return_sequences=True,
        #     return_state=True,
        #     go_backwards=self.bidirectional
        # )
        self.rnn = tf.keras.layers.GRU(
            units=self.output_size,
            return_sequences=True,
            return_state=True,
            go_backwards=False)

    def call(self, inputs, **kwargs):
        outputs = self.rnn(inputs)
        print(outputs)
        if self.rnn_type == "lstm":
            seq_outputs = outputs[0]
            h_states, c_states = [], []
            for num_layer in range(self.num_layers):
                h_states.append(outputs[num_layer+1][0])
                c_states.append(outputs[num_layer+1][1])
            h_state = tf.concat(h_states, axis=-1)
            c_state = tf.concat(c_states, axis=-1)
            return seq_outputs, (h_state, c_state)
        else:
            seq_outputs = outputs[0]
            state = tf.concat(outputs[1:], axis=-1)
            return seq_outputs, (state, None)

if __name__ == "__main__":
    # tf.enable_eager_execution()
    Embedding = ShareEmbeddings(10, 64)
    inputs = tf.constant([[5,3], [1,2]], dtype=tf.int32)
    embed_inputs = Embedding(inputs)
    print(Embedding.trainable_weights)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # grads_and_vars = optimizer.compute_gradients(embed_inputs, Embedding.trainable_variables)
    # for grad in grads_and_vars:
    #     print(grad)
    # grads = tf.clip_by_global_norm(grads_and_vars[1:], clip_norm=5.0)
    # print(grads)
    # optimizer.apply_gradients(zip(grads, Embedding.trainable_variables))
    optimizer.minimize(embed_inputs)
    

    # Encoder = RNNEncoder("gru", hidden_size=128, num_layers=3, bidirectional=False)
    # # print(Encoder.trainable_weights)
    # outputs, states = Encoder(embed_inputs)
    # # print(Encoder.trainable_weights)   # 只有 call 之后才能看到变量，因为 build 函数是在 call 之后才执行。
    # print(outputs.shape, states[0].shape)
    # # for var in Encoder.trainable_weights:
    # #     print(var)
    # print(Encoder.trainable_weights)
