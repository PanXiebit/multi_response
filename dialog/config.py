class Misc(object):
    src_vocab_size = 50000
    tgt_vocab_size = 50000
    tag_vocab_size = 50000
    random_seed = 3435
    use_cuda = True


class Seq2Seq(object):
    rnn_type = "GRU"
    tie_weights = True
    attn_type = "general"
    shared_embedding_size = 620
    tag_hidden_size = 1000
    enc_hidden_size = 1000
    dec_hidden_size = 1000
    num_layers = 1
    dropout = 0.3
    bidirectional = True
    feat_merge = "concat"
    # Seq2Seq Train Opt
    class Trainer:
        num_train_epochs = 10
        steps_per_stats = 100
        batch_size = 256
        optim_method = "adam"
        learning_rate = 0.0002
        max_grad_norm = 5
        learning_rate_decay = 0.5
        weight_decay = 0.000001
        start_decay_at = 0
        out_dir = "./ out / seq2seq"


class TagSampler(object):
    tag_embedding_size = 620
    rnn_type = "GRU"
    enc_hidden_size = 1000
    num_layers = 2
    dropout = 0.3
    bidirectional = True
    # TagSampler Train Opt
    # Trainer
    num_train_epochs = 10
    steps_per_stats = 100
    batch_size = 256
    # start_epoch_at:
    optim_method = "adam"
    learning_rate = 0.002
    max_grad_norm = 5
    learning_rate_decay = 0.5
    weight_decay = 0.000001
    start_decay_at = 0
    out_dir = "./ out / sampler"
    
class Seq2SeqWithRL():
    # Seq2SeqWithRL Train Opt
    # Trainer:
    num_train_epochs = 10
    steps_per_stats = 100
    batch_size = 10
    # start_epoch_at:
    optim_method = "sgd"
    steps_per_update_sampler = 10
    seq2seq_lr = 0.1
    sampler_lr = 0.01
    max_grad_norm = 5
    learning_rate_decay = 0.9
    weight_decay = 0.000001
    start_decay_at = 0
    out_dir = "./ out / rl"
