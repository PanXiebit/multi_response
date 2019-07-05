import tensorflow as tf
from dialog.seq2seq.data_preprocess import Subtokenizer
import codecs
from tqdm import tqdm
import csv

class SeqDataset(object):
    def __init__(self, vocab_file, train_path, vocab_size):
        super(SeqDataset, self).__init__()
        self.subtoken = Subtokenizer.init_from_files(vocab_file, train_path, vocab_size)
        self.vocab_size = self.subtoken.vocab_size

    def convert_string_to_index(self, file_path, ids_file):
        """ convert seq2seq_data_top1.tsv to index file.
        every line have three element. src sentence, tags, and target sentence
        """
        if tf.gfile.Exists(ids_file):
            tf.logging.info("%s is already exists" % ids_file)
        else:
            tf.logging.info("convert string to index, from {} to {}".format(file_path, ids_file))
            with codecs.open(file_path, "r", "utf8") as f, \
                    codecs.open(ids_file, "w", "utf8") as fw:
                csvwrite = csv.writer(fw)
                for i, line in tqdm(enumerate(f)):
                    content = line.strip().split("\t")
                    if len(content) != 3:
                        continue
                    src = content[0]
                    tags = content[1]
                    tgt = content[2]
                    src_ids = " ".join([str(id) for id in self.subtoken.encode(src)])
                    tags_ids = " ".join([str(id) for id in self.subtoken.encode(tags)])
                    tgt_ids = " ".join([str(id) for id in self.subtoken.encode(tgt)])
                    csvwrite.writerow([src_ids, tags_ids, tgt_ids])

    def _parse_train_and_eval_line(self, line):
        record_defaults = [[""] for _ in range(3)]
        src, tag, tgt = tf.decode_csv(line, record_defaults)
        src = self._str2id(src)
        tag = self._str2id(tag)
        tgt = self._str2id(tgt)
        return {"src":src, "tag":tag, "tgt":tgt}

    def _str2id(self, line):
        line = tf.string_split([line]).values
        line = tf.string_to_number(line, tf.int32)
        return line

    def _train_input_fn(self, train_path, train_ids_file,
                        repeat=1, shuffer=True, buffer_size=10000, batch_size=5):
        self.convert_string_to_index(train_path, train_ids_file)
        self.dataset = tf.data.TextLineDataset(train_ids_file)
        dataset = self.dataset.map(self._parse_train_and_eval_line)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                "src": [None],
                "tag": [None],
                "tgt": [None],
            }
        )
        dataset = dataset.repeat(repeat)
        if shuffer:
            dataset = dataset.shuffle(buffer_size)
        iterator = dataset.make_one_shot_iterator()
        dict = iterator.get_next()
        return dict["src"], dict["tag"], dict["tgt"]

    def _eval_input_fn(self, eval_path, eval_ids_file, batch_size=5):
        self.convert_string_to_index(eval_path, eval_ids_file)
        self.dataset = tf.data.TextLineDataset(eval_ids_file)
        dataset = self.dataset.map(self._parse_train_and_eval_line)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        dict = iterator.get_next()["src"]
        return dict["src"], dict["tgt"]


if __name__ == "__main__":
    tf.enable_eager_execution()
    train_path = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/seq2seq_data_top1.tsv"
    vocab_file = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/weibo.vocab.txt"
    train_ids_file = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/seq2seq_data_top1_idx.tsv"
    Dataset = SeqDataset(vocab_file, train_path, vocab_size=50000)
    src, tag, tgt = Dataset._train_input_fn(train_path, train_ids_file)
    print(src.shape, tag.shape, tgt.shape)