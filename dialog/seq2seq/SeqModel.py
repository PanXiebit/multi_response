# -*- encoding=utf8 -*-
import tensorflow as tf
from dialog.seq2seq.modules.JointAttention import JointAttnDecoder


class Seq2SeqWithTag(tf.keras.Model):
    def __init__(self, src_vocab_size, share_embed_size,
                 rnn_type, enc_hidden_size, dec_hidden_size, tag_hidden_size,
                 num_layers, dropout, bidirectional, tie_weights, feat_merge):
        super(Seq2SeqWithTag, self).__init__()
        # embedding layer
        self.src_vocab_size = src_vocab_size
        self.share_embed_size = share_embed_size

        # encoder and decoder, attention parameters
        self.rnn_type = rnn_type
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.tag_hidden_size = tag_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.tie_weights = tie_weights
        self.feat_merge = feat_merge
        
        # embedding layer
        self.Embedding = tf.keras.layers.Embedding(self.src_vocab_size, share_embed_size)

        # tag encoder
        self.TagEncoder = tf.keras.layers.Dense(units=self.tag_hidden_size,
                                                activation=tf.tanh)

        # src encoder
        if self.rnn_type == "gru":
            self.enc_encoder = tf.keras.layers.GRU(units=self.enc_hidden_size,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   go_backwards=bidirectional)
            self.dec_cell = tf.keras.layers.GRUCell(units=dec_hidden_size,
                                                    dropout=dropout)
        elif self.rnn_type == "lstm":
            self.enc_encoder = tf.keras.layers.LSTM(units=self.enc_hidden_size,
                                                    return_sequences=True,
                                                    return_state=True,
                                                    go_backwards=bidirectional)
            self.dec_cell = tf.keras.layers.LSTMCell(units=dec_hidden_size,
                                                     dropout=dropout)

        else:
            raise NameError("rnn typr must be lstm or gru")

        # JointAttentionDecoder
        AttnDecoder = JointAttnDecoder(enc_hidden_size, dec_hidden_size, tag_hidden_size, atten_type="bahdanau")
        
    def call(self, inputs, training=None, mask=None):
        src_inputs = inputs[0]  # [batch, src_len]
        tag_inputs = inputs[1]  # [batch, 1]
        tgt_inputs = inputs[2]  # [batch, tgt_len]
        assert (tag_inputs.shape.as_list()[1] == 1), ("only one tag, and the shape of "
                                                      "tag_input is [batch, 1]")
        src_len = tf.reduce_sum(tf.math.not_equal(src_inputs, 0), axis=1)  # [batch]
        tag_len = tf.reduce_sum(tf.math.not_equal(tag_inputs, 0), axis=1)  # [batch]
        tgt_len = tf.reduce_sum(tf.math.not_equal(tgt_inputs, 0), axis=1)  # [batch]
        
        
        

    def tag_encoder(self, tag_inputs):
        tag_embed = self.Embedding(tag_inputs)
        tag_hidden = self.TagEncoder(tag_embed)
        return tag_hidden

    def src_encoder(self, src_inputs):
        src_emb = self.Embedding(src_inputs)
        if self.rnn_type == "gru":
            outputs = self.enc_encoder(src_emb)
            encoder_outputs, enc_hidden = outputs
        else:
            outputs = self.enc_encoder(src_emb)
            encoder_outputs = outputs[0]
            enc_hidden = outputs[1:]
        return encoder_outputs, enc_hidden

    def init_decoder_state(self, enc_hidden, tag_hidden):
        """
        
        :param enc_hidden:  [batch, enc_hidden_size]
        :param tag_hidden:  [batch, tah_hidden_size]
        :return: 
        """
        if not isinstance(enc_hidden, tuple):  # GRU
            if self.feat_merge == "sum":
                h = enc_hidden + tf.broadcast_to(tag_hidden, tf.shape(enc_hidden))
            elif self.feat_merge == "concat":
                h = tf.concat([enc_hidden, tf.broadcast_to(tag_hidden, tf.shape(enc_hidden))], axis=-1)
            else:
                raise NameError("feat merge must be sum or concat.")
            return h
        else:  # LSTM
            h, c = enc_hidden
            tag_hidden = tf.broadcast_to(tag_hidden, tf.shape(enc_hidden))
            if self.feat_merge == "sum":
                h += tag_hidden
                c += tag_hidden
            elif self.feat_merge == "concat":
                h = tf.concat([h, tag_hidden], axis=-1)
                c = tf.concat([c, tag_hidden], axis=-1)
            return h,c
        
    def decoder(self, rnn_cell, tgt_inputs, tag_hidden, enc_outputs, decoder_init_states, training=None):
        """
        
        :param tgt_inputs:
        :param tag_hidden:
        :param enc_outputs:
        :param decoder_init_states:
        :param training:
        :return:
        """
        if training:
            pass
                
             

    





