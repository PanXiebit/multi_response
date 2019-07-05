from __future__ import print_function
import yaml
import codecs

def load_config(config_file):
    with codecs.open(config_file, 'r', encoding='utf8') as f:
        config = yaml.load(f)
        return config