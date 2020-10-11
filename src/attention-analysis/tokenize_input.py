import json

import numpy as np
import tensorflow as tf
from transformers import BartTokenizer


def load_json(path):
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def tokenize(model_dir, data_file, cased=False, max_sequence_length=512):
    tokenizer = BartTokenizer.from_pretrained(model_dir, do_lower_case=not cased)

    examples = []
    for features in load_json(data_file):
        example = Example(features, tokenizer, max_sequence_length)
        if len(example.input_ids) <= max_sequence_length:
            examples.append(example)

    return examples


class Example(object):
    """Represents a single input sequence to be passed into BERT."""

    def __init__(self, features, tokenizer, max_sequence_length):
        self.features = features

        if "tokens" in features:
            self.tokens = tokenizer.convert_tokens_to_ids(features["tokens"])
        else:
            if "text" in features:
                text = features["text"]
            else:
                text = " ".join(features["words"])
            self.tokens = tokenizer.encode_plus(text,
                                                add_special_tokens=True,
                                                max_length=max_sequence_length,
                                                pad_to_max_length=True)

        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.segment_ids = [0] * len(self.input_ids)
        self.input_mask = [1] * len(self.input_ids)
        while len(self.input_ids) < max_sequence_length:
            self.input_ids.append(0)
            self.input_mask.append(0)
            self.segment_ids.append(0)
