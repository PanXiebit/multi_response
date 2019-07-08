from __future__ import division


import tensorflow as tf
from tensorflow.python.util import nest


class JointAttnDecoder(tf.keras.layers.Layer):
    def __init__(self, rnn_cell, enc_hidden_size, dec_hidden_size, tag_hidden_size, 
                 output_size, attn_type="bahdanau"):
        super(JointAttnDecoder, self).__init__()
        self.rnn_cell = rnn_cell
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.tag_hidden_size = tag_hidden_size
        assert enc_hidden_size == dec_hidden_size
        
        self.output_size = output_size
        self.attn_type = attn_type
        assert (self.attn_type in ["bahdanau", "luong"]), (
            "Please select a valid attention type.")
        if self.attn_type == "bahdanau":
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
        if self.attn_type == "bahdanau":
            attn_score = tf.reduce_sum(self.src_V(tf.tanh(tf.expand_dims(self.src_W(h_t), axis=1) +
                                       self.src_U(enc_hidden))), axis=2, keep_dims=False)          # [batch, src_len]
            # mask_score

            if enc_length is None:
                tf.logging.info("Don't mask score")
                attn_score = attn_score
            else:
                score_mask = tf.sequence_mask(lengths=enc_length, maxlen=tf.shape(attn_score)[1])  # [batch, src_len]
                score_mask_values = tf.ones_like(attn_score, dtype=tf.float32) * score_mask_value
                attn_score = tf.where(score_mask, attn_score, score_mask_values)

        # smooth
        attn_score /= tf.sqrt(float(self.enc_hidden_size) + 1e-6)
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
        
        if self.attn_type == "bahdanau":
            attn_score = tf.reduce_sum(self.tag_V(tf.tanh(tf.expand_dims(self.tag_W(tag_hidden), axis=1) +
                                        self.tag_U(enc_outputs))), axis=2, keep_dims=False) # [batch, src_len]
            if enc_length is None:
                tf.logging.info("Don't mask score")
                attn_score = attn_score
            else:
                score_mask = tf.sequence_mask(lengths=enc_length, maxlen=tf.shape(attn_score)[1])  # [batch, src_len]
                score_mask_values = tf.ones_like(attn_score, dtype=tf.float32) * score_mask_value
                attn_score = tf.where(score_mask, attn_score, score_mask_values)

        # smooth
        attn_score /= tf.sqrt(float(self.enc_hidden_size) + 1e-6)
        attn_weights = tf.nn.softmax(attn_score)  # [batch, src_len]
        context_vec = tf.reduce_sum(
            tf.expand_dims(attn_weights, axis=-1) * enc_hidden, axis=1)  # [batch, enc_hidden_size]
        return context_vec

    def step(self, cur_tgt_input, prev_state, tag_hidden, enc_outputs, enc_length):
        """ one step rnn decoder, with tags.
        
        :param cur_tgt_input: [batch, tgt_embed_size]
        :param prev_state : [batch, dec_hidden_szie]
        :param tag_hidden: [batch, tag_hidden_size]
        :param enc_outputs:  [batch, src_len, enc_hidden_size]
        :param enc_length: [batch]
        :return: 
        """
        tgt_hidden, next_state = self.rnn_cell(cur_tgt_input, [prev_state])
        tgt_context_vec = self.comp_tgt_context_vec(tgt_hidden, enc_outputs, enc_length)  # [batch, dec_hidden_size]
        tag_context_vec = self.comp_tag_context_vec(tag_hidden, enc_outputs, enc_length)
        # concatenate
        concat_c = tf.concat([tgt_context_vec, tag_context_vec, tgt_hidden], axis=1)
        out = self.linear_out(concat_c)   # [batch, dec_hidden_size]
        return out, next_state[0]

    def train_decoder(self, tgt_inputs, tgt_length, tag_hidden, enc_outputs, enc_length,
                     decoder_init_states, maximum_iteration=None):
        """ Perform training dynamic decoding with `step`.
        Calls initialize() once and step() repeatedly on the Decoder object.

        :param tgt_inputs: [batch, tgt_max_times, dec_hidden_size]
        :param tgt_length: [batch]
        :param tag_hidden: [batch, tag_hidden_size]
        :param enc_outputs: [batch, src_max_times, enc_hidden_size]
        :param decoder_init_states: [batch, enc_hidden_size]
        :param maximum_iteration: 在训练阶段是 None
        :param training:
        :return:
        """

        def _unstack_ta(inp):
            return tf.TensorArray(
                dtype=inp.dtype, size=tf.shape(inp)[0],
                element_shape=inp.get_shape()[1:]).unstack(inp)

        dynamic_size = maximum_iteration is None
        batch_size = tgt_inputs.get_shape().as_list()[0]

        init_time = tf.constant(0, dtype=tf.int32)
        initial_outputs_ta = tf.TensorArray(dtype=tf.float32,
                                            size=0,
                                            dynamic_size=dynamic_size,
                                            element_shape=(batch_size, self.output_size))
        # time major
        tgt_inputs = tf.transpose(tgt_inputs, (1, 0, 2))  # [tgt_max_times, batch, dec_hidden_size]
        self._zero_inputs = nest.map_structure(
            lambda inp: tf.zeros_like(inp[0, :]), tgt_inputs)

        tgt_inputs_tas  = nest.map_structure(_unstack_ta, tgt_inputs)

        init_state = decoder_init_states
        # init_inputs = tgt_inputs[init_time]
        init_inputs = tgt_inputs_tas.read(0)
        init_finished = tf.equal(0, tf.convert_to_tensor(tgt_length, dtype=tf.int32))   # 初始为False

        def condition(unsed_time, unused_outputs_ta, unsed_state, unsed_inputs, finished):
            return tf.logical_not(tf.reduce_all(finished)) # tf.logical_all 只要有一个 False，它就为False

        def body(time, outputs_ta, state, inputs, finished):
            """Internal while_loop body.
             Args:
               time: scalar int32 tensor.
               outputs_ta: structure of TensorArray.
               state: (structure of) state tensors and TensorArrays.
               inputs: (structure of) input tensors.
               finished: bool tensor (keeping track of what's finished).
             Returns:
               `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                 next_sequence_lengths)`.
               ```
             """
            next_time = time + 1
            next_output, next_state = self.step(
                tgt_inputs[time], state, tag_hidden, enc_outputs, enc_length)
            outputs_ta = outputs_ta.write(time, next_output)
            # next_input = tgt_inputs[next_time]
            finished = (next_time >= tgt_length)
            all_finished = tf.reduce_all(finished) # 这里的all_finish 必须是计算next time的，用来在最后一个time step选择zero_inputs
            def read_from_ta(inp):
                return inp.read(next_time)
            next_inputs = tf.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(read_from_ta, tgt_inputs_tas))
            # next_input = tgt_inputs_tas.read(next_time) # 在最后一个时间步的时候会报错，因为最后一个time step的下一个input是没有的
            return next_time, outputs_ta, next_state, next_inputs, finished

        # tf.while_loop:
        # `cond` and `body` both take as many arguments as there are `loop_vars`.
        res = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=(
                init_time,
                initial_outputs_ta,
                init_state,
                init_inputs,
                init_finished
            ),
            parallel_iterations = 32,
            maximum_iterations = maximum_iteration,
            swap_memory = False
        )
        final_outputs_ta = res[1]
        final_state = res[2]
        final_outputs = tf.transpose(final_outputs_ta.stack(), (1,0,2))

        return final_outputs, final_state

    def infer_decoder(self, embedding, start_token, end_token):
        """
        :param embedding:
        :return:
        """
        pass


if __name__ == "__main__":
    # tf.enable_eager_execution()
    rnncell = tf.keras.layers.GRUCell(units=32)
    num_units = 32
    attndecoder = JointAttnDecoder(rnncell,
                                   enc_hidden_size=num_units,
                                   dec_hidden_size=num_units,
                                   tag_hidden_size=num_units,
                                   output_size=num_units)

    batch_size = 5
    tgt_max_times = 6
    src_max_times = 6
    tgt_length = [3,5,6,4,3]
    enc_length = [5,6,5,3,4]
    decoder_init_states = tf.zeros((batch_size, num_units), tf.float32)
    tgt_inputs  = tf.random_normal((batch_size, tgt_max_times, num_units), dtype=tf.float32)
    enc_outputs = tf.random_normal((batch_size, src_max_times, num_units), dtype=tf.float32)
    tag_hidden = tf.random_normal((batch_size, num_units), dtype=tf.float32)
    out = attndecoder.train_decoder(tgt_inputs, tgt_length, tag_hidden, enc_outputs, enc_length,
                     decoder_init_states, maximum_iteration=None)
    print(out)
    for var in attndecoder.trainable_weights:
        print(var)