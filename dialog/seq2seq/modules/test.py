import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.ops import rnn_cell
#
# tf.enable_eager_execution()
batch_size = 5
src_len = [4,5,3,5,6]
max_times = 6
num_units = 16
enc_output = tf.random.normal((batch_size, max_times, num_units), dtype=tf.float32)
#
# attenRNNCell
rnncell = rnn_cell.LSTMCell(num_units=16)
attention_mechanism = attention_wrapper.BahdanauAttention(
    num_units=num_units,
    memory=enc_output,
    memory_sequence_length=src_len)
attnRNNCell= attention_wrapper.AttentionWrapper(
    cell=rnncell,
    attention_mechanism=attention_mechanism,
    alignment_history=True)

# training
tgt_len = [5,6,2,7,4]
tgt_max_times = 7
tgt_inputs = tf.random.normal((batch_size, tgt_max_times, num_units), dtype=tf.float32)
training_helper = helper_py.TrainingHelper(tgt_inputs, tgt_len)

# train helper
train_decoder = basic_decoder.BasicDecoder(
    cell=attnRNNCell,
    helper=training_helper,
    initial_state=attnRNNCell.zero_state(batch_size, tf.float32)
)

# inference
embedding = tf.get_variable("embedding", shape=(10, 16), initializer=tf.random_uniform_initializer())
infer_helper = helper_py.GreedyEmbeddingHelper(
    embedding=embedding, # 可以是callable，也可以是embedding矩阵
    start_tokens=tf.zeros([batch_size], dtype=tf.int32),
    end_token=9
)
infer_decoder = basic_decoder.BasicDecoder(
    cell=attnRNNCell,
    helper=infer_helper,
    initial_state=attnRNNCell.zero_state(batch_size, tf.float32)
)
final_outputs, final_state, final_sequence_lengths = decoder.dynamic_decode(
    train_decoder,
    maximum_iterations=False)

print(final_outputs.rnn_output)
print(final_outputs.sample_id)
print( final_state.cell_state)
print(final_sequence_lengths)

print("--------infer-------------")
final_outputs, final_state, final_sequence_lengths = decoder.dynamic_decode(
    infer_decoder,
    maximum_iterations=False)

print(final_outputs.rnn_output)
print(final_outputs.sample_id)
print( final_state.cell_state)
print(final_sequence_lengths)