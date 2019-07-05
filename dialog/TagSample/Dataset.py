import tensorflow as tf
from dialog.TagSample.data_preprocess import Subtokenizer
import codecs
from tqdm import tqdm
import csv

class get_dataset(object):
    def __init__(self, vocab_file, train_path, vocab_size):
        super(get_dataset, self).__init__()
        self.subtoken = Subtokenizer.init_from_files(vocab_file, train_path, vocab_size)
        self.vocab_size = self.subtoken.vocab_size

    def convert_string_to_index(self, file_path, ids_file):
        if tf.gfile.Exists(ids_file):
            tf.logging.info("%s is already exists" % ids_file)
        else:
            tf.logging.info("convert string to index, from {} to {}".format(file_path, ids_file))
            with codecs.open(file_path, "r", "utf8") as f, \
                    codecs.open(ids_file, "w", "utf8") as fw:
                csvwrite = csv.writer(fw)
                for i, line in tqdm(enumerate(f)):
                    content = line.strip().split("\t")
                    if len(content) != 2:
                        continue
                    src = content[0]
                    tgt = content[1]
                    src_ids = " ".join([str(id) for id in self.subtoken.encode(src)])
                    tgt_ids = " ".join([str(id) for id in self.subtoken.encode(tgt)])
                    csvwrite.writerow([src_ids, tgt_ids])

    def _parse_train_and_eval_line(self, line):
        record_defaults = [[""] for _ in range(2)]
        src, tgt = tf.decode_csv(line, record_defaults)
        src = self._str2id(src)
        tgt = self._str2id(tgt)
        return {"src":src, "tgt":tgt}

    def _str2id(self, line):
        line = tf.string_split([line]).values
        line = tf.string_to_number(line, tf.int32)
        return line

    def _train_input_fn(self, train_path, train_ids_file, batch_size,
                        repeat=1, shuffer=True, buffer_size=10000):
        self.convert_string_to_index(train_path, train_ids_file)
        self.dataset = tf.data.TextLineDataset(train_ids_file)
        dataset = self.dataset.map(self._parse_train_and_eval_line)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                "src": [None],
                "tgt": [None],
            }
        )
        dataset = dataset.repeat(repeat)
        if shuffer:
            dataset = dataset.shuffle(buffer_size)
        iterator = dataset.make_one_shot_iterator()
        dict = iterator.get_next()
        return dict["src"], dict["tgt"]

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
    train_path = "/home/panxie/Documents/myGAN/multi-response/data/weibo/sampler_data.tsv"
    vocab_file = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/weibo.vocab.txt"
    train_ids_file = "/home/panxie/Documents/myGAN/tf_multi_response/data/weibo/sampler_data_idx.csv"
    Dataset = get_dataset(vocab_file, train_path)
    src, tgt = Dataset._train_input_fn(train_path, train_ids_file)
    print(src.shape, tgt.shape)