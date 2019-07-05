from __future__ import division


import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.decoder import dynamic_decode, Decoder
from tensorflow.python.util import nest


class JointAttnDecoder(tf.keras.layers.Layer):
    def __init__(self, rnn_cell, enc_hidden_size, dec_hidden_size, tag_hidden_size, 
                 output_size, atten_type="bahdanau"):
        super(JointAttnDecoder, self).__init__()
        self.rnn_cell = rnn_cell
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.tag_hidden_size = tag_hidden_size
        assert enc_hidden_size == dec_hidden_size
        
        self.output_size = output_size
        self.atten_type = atten_type
        assert (self.attn_type in ["bahdanau", "luong"]), (
            "Please select a valid attention type.")
        if self.atten_type == "bahdanau":
            self.src_W = tf.keras.layers.Dense(units=dec_hidden_size)
            self.src_U = tf.keras.layers.Dense(units=dec_hidden_size)
            self.src_V = tf.keras.layers.Dense(units=dec_hidden_size)
            self.tag_W = tf.keras.layers.Dense(units=dec_hidden_size)
            self.tag_U = tf.keras.layers.Dense(units=dec_hidden_size)
            self.tag_V = tf.keras.layers.Dense(units=dec_hidden_size)
        else:
            self.src_linear = tf.keras.layers.Dense(units=dec_hidden_size)
            self.tgt_linear = tf.keras.layers.Dense(units=dec_hidden_size)
        self.linear_out = tf.keras.layers.Dense(units=output_size, activation=tf.tanh)

    def comp_tgt_context_vec(self, h_t, enc_hidden, enc_length, score_mask_value=-float("inf")):
        """ computer source-side context vector ct with target current hidden state

        :param h_t:  current target hidden state, [batch, hidden_size]
        :param enc_hidden: encoder hidden state, [batch, src_len, hidden_size]
        :param enc_length: encoder sequence length, [batch]
        :return: context vector, [batch, ]
        """
        # luong attention
        if self.atten_type == "bahdanau":
            attn_score = tf.reduce_sum(self.src_V(tf.tanh(tf.expand_dims(self.src_W(h_t), axis=1) +
                                       self.src_U(enc_hidden))), axis=2, keep_dims=False)          # [batch, src_len]
            # mask_score

            if enc_length is None:
                tf.logging.info("Don't mask score")
                attn_score = attn_score
            else:
                score_mask = tf.sequence_mask(lengths=enc_length, maxlen=tf.shape(attn_score)[1])  # [batch, src_len]
                score_mask_values = tf.cast(score_mask_value, tf.float32)
                attn_score = tf.where(score_mask, attn_score, score_mask_values)

        # smooth
        attn_score /= tf.sqrt(self.enc_hidden_size)
        attn_weights = tf.nn.softmax(attn_score) # [batch, src_len]
        context_vec = tf.reduce_sum(
            tf.expand_dims(attn_weights, axis=-1) * enc_hidden, axis=1)   # [batch, enc_hidden_size]
        return context_vec

    def comp_tag_context_vec(self, tag_hidden, enc_hidden, enc_length, score_mask_value=-float("inf")):
        """ computer source-side context vector ct with tag hidden state

        :param tag_hidden:  [batch, tag_len, tag_hidden_size]
        :param enc_hidden:  [batch, src_len, enc_hidden_size]
        :return:
        """
        assert (tag_hidden.shape.ndims == 2), ("every example have only one tag. so "
                                               "tag_hidden.shape=(batch, tah_hidden_size)"
                                               "instead of {}".format(tag_hidden.shape))
        
        if self.atten_type == "bahdanau":
            attn_score = tf.reduce_sum(self.tag_V(tf.tanh(tf.expand_dims(self.tag_W(tag_hidden), axis=1) +
                                        self.tag_U(tag_hidden))), axis=2, keep_dims=False) # [batch, src_len]
            if enc_length is None:
                tf.logging.info("Don't mask score")
                attn_score = attn_score
            else:
                score_mask = tf.sequence_mask(lengths=enc_length, maxlen=tf.shape(attn_score)[1])  # [batch, src_len]
                score_mask_values = tf.cast(score_mask_value, tf.float32)
                attn_score = tf.where(score_mask, attn_score, score_mask_values)

        # smooth
        attn_score /= tf.sqrt(self.enc_hidden_size)
        attn_weights = tf.nn.softmax(attn_score)  # [batch, src_len]
        context_vec = tf.reduce_sum(
            tf.expand_dims(attn_weights, axis=-1) * enc_hidden, axis=1)  # [batch, enc_hidden_size]
        return context_vec

    def step(self, cur_time, cur_tgt_input, prev_state, tag_hidden, enc_outputs, enc_length, training=None):
        """ one step rnn decoder, with tags.
        
        :param cur_tgt_input: [batch, tgt_embed_size]
        :param prev_state : [batch, dec_hidden_szie]
        :param tag_hidden: [batch, tag_hidden_size]
        :param enc_outputs:  [batch, src_len, enc_hidden_size]
        :param enc_length: [batch]
        :return: 
        """
        tgt_hidden, next_state = self.rnn_cell(cur_tgt_input, prev_state)
        tgt_context_vec = self.comp_tgt_context_vec(tgt_hidden, enc_outputs, enc_length)  # [batch, dec_hidden_size]
        tag_context_vec = self.comp_tag_context_vec(tag_hidden, enc_outputs, enc_length)
        # concatenate
        concat_c = tf.concat([tgt_context_vec, tag_context_vec, tgt_hidden], axis=1)
        out = self.linear_out(concat_c)   # [batch, dec_hidden_size]
        return cur_time + 1, out, next_state

    def call(self, tgt_inputs, tag_hidden, enc_outputs, decoder_init_states,
             max_length, training=None):
        """ Perform dynamic decoding with `step`.
        Calls initialize() once and step() repeatedly on the Decoder object.

        :param tgt_inputs: [batch, target_len, tgt_hidden_size]
        :param tag_hidden: [batch, tag_hidden_size]
        :param enc_outputs: [batch, src_len, src_hidden_size]
        :param decoder_init_states: [batch, dec_hidden_size]
        :param max_length:
        :param training:
        :return:
        """
        dynamic_size = max_length is None
        batch_size = tf.shape(enc_outputs)[0]

        def _create_ta(size):
            return tf.TensorArray(
                dtype=tf.float32,
                size=0 if dynamic_size else max_length,
                dynamic_size=dynamic_size,
                element_shape=(batch_size, size))

        initial_outputs_ta = nest.map_structure(_create_ta, self.output_size) # map_structure 与 map 的区别在于其返回的就是与
        init_time = 0
        init_state = decoder_init_states
        init_finished = 

        def body(time, outputs_ta, state, inputs, finished, sequence_lens):
            """Internal while_loop body.
             Args:
               time: scalar int32 tensor.
               outputs_ta: structure of TensorArray.
               state: (structure of) state tensors and TensorArrays.
               inputs: (structure of) input tensors.
               finished: bool tensor (keeping track of what's finished).
               sequence_lengths: int32 tensor (keeping track of time of finish).
             Returns:
               `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                 next_sequence_lengths)`.
               ```
             """
            (next_outputs, decoder_state, next_inputs, de)


if __name__ == "__main__":
    def _create_ta(size):
        return tf.TensorArray(
            dtype=tf.float32,
            size=10,
            dynamic_size=True,
            element_shape=(5, size))


    initial_outputs_ta = nest.map_structure(_create_ta, 32)
    print(initial_outputs_ta)