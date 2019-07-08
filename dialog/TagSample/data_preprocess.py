# -*- encoding=utf8 -*-

""" Subtokenizer class to encode and decode strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import codecs
import numpy as np
import six
import tensorflow as tf
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

PAD = "<pad>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1
UNK = "UNK"
UNK_ID = 2
RESERVED_TOKENS = [PAD, EOS, UNK]

def get_word_count(filename):
    word_count = collections.defaultdict(int)
    with codecs.open(filename, "r", "utf8") as f:
        for i, line in tqdm(enumerate(f)):
            #if i > 5:
            #    break
            content = line.strip().split("\t")
            if len(content) != 2:
                continue
            src_text = content[0]
            tags_text = content[1]
            for word in src_text.split() + tags_text.split():
                word_count[word] += 1
        return word_count


def save_vocab_file(filename, vocab_file, vocab_size):
    if not tf.gfile.Exists(vocab_file):
        tf.logging.info("get vocab from %s" %filename)
        word_count = get_word_count(filename)
        word_count_pairs = collections.Counter(word_count).most_common(vocab_size)
        with codecs.open(vocab_file, "w", "utf8") as fw:
            for word, _ in word_count_pairs:
                fw.write(word + "\n")


class Subtokenizer(object):
    """ 如果使用了 bpe，则 '_split_token_to_subtokens' 这个函数是有意义的。
        如果没有用 bpe，以 word 为单位，则这个函数没啥用。但用在这里也没啥影响。
    """
    def __init__(self, vocab_file, train_path, reserved_tokens=None):
        """Initializes class, creating a vocab file if data_files is provided."""
        tf.logging.info("Initializing Subtokenizer from file %s." % vocab_file)
        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS
        # 根据词表文件获取词表
        self.subtoken_list, self.vocab_size = _load_vocab_file(vocab_file, reserved_tokens)
        # 根据词表获取字典
        self.subtoken_to_id_dict = {item: n for n, item in enumerate(self.subtoken_list)}

    @staticmethod
    def init_from_files(vocab_file, train_path, vocab_size):
        """初始化Subtokenizer. 如果不存在词表文件，则需要另外加载。
        """
        if not tf.gfile.Exists(vocab_file):
            tf.logging.info("Vocab file not exists (%s), and load vocab from %s" % (vocab_file, train_path))
            save_vocab_file(train_path, vocab_file, vocab_size)
        else:
            tf.logging.info("Vocab file already exists (%s)" % vocab_file)
        return Subtokenizer(vocab_file, train_path)

    def encode(self, raw_string, add_eos=False):
        """Encodes a string into a list of int subtoken ids."""
        ret = []
        tokens = raw_string.strip().split()
        for token in tokens:
            ret.extend(self._token_to_ids(_native_to_unicode(token)))
        if add_eos:
            ret.append(EOS_ID)
        return ret

    def _token_to_ids(self, token):
        """Encode a single token into a list of subtoken ids."""
        ret = _split_token_to_subtokens(token, self.subtoken_to_id_dict)
        ret = [self.subtoken_to_id_dict[subtoken_id] for subtoken_id in ret]
        return ret

    def decode(self, subtokens):
        """Converts list of int subtokens ids into a string."""
        if isinstance(subtokens, np.ndarray):
            # Note that list(subtokens) converts subtokens to a python list, but the
            # items remain as np.int32. This converts both the array and its items.
            subtokens = subtokens.tolist()
        if not subtokens:
            return ""

        assert isinstance(subtokens, list) and isinstance(subtokens[0], int), (
            "Subtokens argument passed into decode() must be a list of integers.")
        return _unicode_to_native(
            " ".join(self._ids_to_tokens(subtokens)))

    def _ids_to_tokens(self, subtokens):
        """Convert list of int subtoken ids to a list of string tokens."""
        ret = []
        for s in subtokens:
            # 根据token的index来直接转换成 token
            ret.append(self.subtoken_list[s])
        return ret

def _load_vocab_file(vocab_file, reserved_tokens=None):
    if reserved_tokens is None:
        reserved_tokens = RESERVED_TOKENS
    with tf.gfile.Open(vocab_file, mode="r") as f:
        subtoken_list = f.readlines()
    reserved_tokens = [_native_to_unicode(word) for word in reserved_tokens]
    subtoken_list = [_native_to_unicode(word.strip()) for word in subtoken_list ]
    subtoken_list =[word for word in subtoken_list if word not in reserved_tokens]
    subtoken_list = reserved_tokens +subtoken_list
    tf.logging.info("total vocabulary size:{}".format(len(subtoken_list)))
    return subtoken_list, len(subtoken_list)


def _native_to_unicode(s):
    """Convert string to unicode (required in Python 2)."""
    if six.PY2:
        return s if isinstance(s, unicode) else s.decode("utf-8")
    else:
        return s

def _unicode_to_native(s):
    """Convert string from unicode to native format (required in Python 2)."""
    if six.PY2:
        return s.encode("utf-8") if isinstance(s, unicode) else s
    else:
        return s

def _split_token_to_subtokens(token, subtoken_dict):
    """Splits a token into subtokens defined in the subtoken dict."""
    ret = []
    if token in subtoken_dict:
        ret.append(token)
    else:
        ret.append(_native_to_unicode(UNK))
    return ret


if __name__ == "__main__":
    train_path = "/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/sampler_data.tsv"
    vocab_file = "/home/work/xiepan/xp_dial/tf_multi_response/data/weibo/weibo.vocab_2.txt"
    subtoken = Subtokenizer.init_from_files(vocab_file, train_path, 50000)
    text_str = "哈哈 嘻嘻 谢谢 你好 我 我们"
    print(subtoken.encode(text_str, add_eos=True))




