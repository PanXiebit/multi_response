# -*- encoding=utf8 -*- 
import argparse
import dialog.utils as utils
import tensorflow as tf
import config
#from tensorflow_estimator import estimator

#tf.enable_eager_execution()

import os
from dialog.TagSample.model import TagSimple
from dialog.TagSample.Dataset import get_dataset
#from dialog.TagSample.modelHelper import ShareEmbeddings, RNNEncoder
from dialog.TagSample.loss import SamplerLossCompute
tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
#parser.add_argument("-config",
#                    default="/home/work/xiepan/xp_dial/tf_multi_response/config.yml",
#                    type=str)
parser.add_argument("-vocab",
                    default="/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/weibo.vocab.txt",
                    type=str)
parser.add_argument('-train_data',
                    default="/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/sampler_data.tsv",
                    type=str)
parser.add_argument('-train_ids_file',
                    default="/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/sampler_data_idx.csv",
                    type=str)
parser.add_argument('-eval_data',
                    default="/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/sampler_data.tsv",
                    type=str)
parser.add_argument('-eval_ids_file',
                    default="/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/sampler_data_idx.csv",
                    type=str)
parser.add_argument("-log_dir",
                    default="/home/work/xiepan/xp_dial/tf_multi_response/out/log_dir",
                    type=str)
parser.add_argument("-model_dir",
                    default="/home/work/xiepan/xp_dial/tf_multi_response/out/model_save",
                    type=str)

args = parser.parse_args()


# 定义训练集和验证集
my_dataset = get_dataset(args.vocab, args.train_data, config.Misc.src_vocab_size)
def train_input_fn():
    return my_dataset._train_input_fn(args.train_data, args.train_ids_file, batch_size=config.TagSampler.batch_size)

def eval_input_fn():
    return my_dataset._eval_input_fn(args.eval_data, args.eval_ids_file, batch_size=config.TagSampler.batch_size)

# 定义模型框架
loss_obj = SamplerLossCompute()

def model_fn(features, labels, mode):
    # define model, model_1
    base_model = TagSimple(vocab_size=config.Misc.src_vocab_size, 
                           embedding_size=config.TagSampler.tag_embedding_size, 
                           hidden_size=config.TagSampler.enc_hidden_size, 
                           rnn_type=config.TagSampler.rnn_type,
                           bidirectional=config.TagSampler.bidirectional)
    # model_2
    #base_model = TagSimple(my_dataset.vocab_size, embedding_size=64, hidden_size=100, rnn_type="lstm")
    log_prob = base_model(features)
    #log_prob_shape = log_prob.shape
    tf.identity(log_prob, "predictions")
    
    # prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=log_prob)

    # train and evaluation
    loss = loss_obj(log_prob, labels)
    tf.identity(loss, "loss")

    TENSORS_TO_LOG = {
        "loss": loss}
        #"predictions": log_prob_shape}

    logging_hook = tf.train.LoggingTensorHook(
        TENSORS_TO_LOG,
        every_n_iter=1000)

    if mode == tf.estimator.ModeKeys.TRAIN:
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        grads_and_vars = optimizer.compute_gradients(loss, base_model.trainable_variables)
        # grads = tf.clip_by_global_norm(grads_and_vars, clip_norm=5.0)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step(), name="train_op")
        # train_op = optimizer.minimize(loss)
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=log_prob,
            loss=loss,
            train_op=train_op,
            training_hooks=[logging_hook])
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=log_prob,
            loss=loss, 
            training_hooks=[logging_hook])


# setup train spec
train_spec = tf.estimator.TrainSpec(
    input_fn=lambda:train_input_fn(),
    max_steps=30000)
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: eval_input_fn(),
    steps=200)

def train_and_eval():
    esti_tagsimple = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(args.model_dir, "train"))

    tf.estimator.train_and_evaluate(esti_tagsimple, train_spec, eval_spec)


if __name__ == "__main__":
    train_and_eval()
