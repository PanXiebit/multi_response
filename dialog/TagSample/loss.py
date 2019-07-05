from __future__ import division

import tensorflow as tf

class SamplerLossCompute(tf.keras.layers.Layer):
    def __init__(self):
        super(SamplerLossCompute, self).__init__()


    def call(self, log_prob, tags_label):
        """
        
        :param log_prob:  [batch, tag_vocab_size] 
        :param tags_label:  [batch, tags_len]
        :return: 
        """
        tags_mask = tf.cast(tf.math.not_equal(tags_label, 0), tf.float32)  # []
        print("log_prob.shape", log_prob.shape)
        print("label.shape", tags_label.shape)
        loss = tf.batch_gather(log_prob, tags_label)    # [batch, tags_len]
        print("loss.shape", loss.shape)
        loss = loss * tags_mask
        loss = - tf.reduce_mean(loss)
        return loss

if __name__ == "__main__":
    tf.enable_eager_execution()
    loss_obj = SamplerLossCompute()
    log_prob = tf.random_normal(shape=[2, 5000])
    tags_label = tf.constant([[3,2,1,0,0], [6,7,8,9,10]])
    loss = loss_obj(log_prob, tags_label)
    print(loss)

        
