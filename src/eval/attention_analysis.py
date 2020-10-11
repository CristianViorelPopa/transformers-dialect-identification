import os
import pickle
import argparse

import matplotlib
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

sns.set_style("darkgrid")

width = 3
example_sep = 3
word_height = 1
pad = 0.1
words_occurrences = {}


"""
NOTE: The code in this file is, for the most part, taken from the notebooks in 
https://github.com/clarkkev/attention-analysis, namely `General_Analysis.ipynb`
"""


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding="latin1")


def data_iterator(data):
    for i, doc in enumerate(data):
        if i % 100 == 0 or i == len(data) - 1:
            print("{:.1f}% done".format(100.0 * (i + 1) / len(data)))
        yield doc["tokens"], np.array(doc["attns"])


def compute_max_len(example):
    """Compute the max length of the sequence, before the padding tokens"""
    if '[PAD]' not in example['tokens']:
        return len(example['tokens'])
    return example['tokens'].index('[PAD]')


def get_data_points(head_data):
      xs, ys, avgs = [], [], []
      for layer in range(12):
            for head in range(12):
                ys.append(head_data[layer, head])
                xs.append(1 + layer)
            avgs.append(head_data[layer].mean())
      return xs, ys, avgs


def get_example_attn(example, heads):
    """Plots attention maps for the given example and attention heads."""
    max_len = compute_max_len(example)

    # if heads is an empty list, we instead output the average attention over all the layers and heads
    if not heads:
        attn = np.zeros(shape=(max_len, max_len))
        for layer in range(len(example['attns'])):
            # attn = example["attns"][layer][head][:max_len,:max_len]
            for head_idx in range(len(example['attns'][layer])):
                attn += example["attns"][layer][head_idx][:max_len, :max_len]

        # terms_attn = np.zeros(shape=(max_len))
        # for term in range(max_len)):
        #     terms_attn[term] = attn[:, ]
        terms_attn = np.sum(attn, axis=0)
        words = np.array(example["tokens"][:max_len])
        for word in words:
            if word not in words_occurrences:
                words_occurrences[word] = 0
            words_occurrences[word] += 1

        return dict(zip(words, terms_attn))


def add_line(avg_attns, key, ax, color, label, plot_avgs=True):
      xs, ys, avgs = get_data_points(avg_attns[key])
      ax.scatter(xs, ys, s=12, label=label, color=color)
      if plot_avgs:
          ax.plot(1 + np.arange(len(avgs)), avgs, color=color)
      ax.legend(loc="best")
      ax.set_xlabel("Layer")
      ax.set_ylabel("Avg. Attention")


def plot_entropies(xs, uniform_attn_entropy, ax, data, avgs, label, c):
      ax.scatter(xs, data, c=c, s=5, label=label)
      ax.plot(1 + np.arange(12), avgs, c=c)
      ax.plot([1, 12], [uniform_attn_entropy, uniform_attn_entropy],
              c="k", linestyle="--")
      ax.text(7, uniform_attn_entropy - 0.45, "uniform attention",
              ha="center")
      ax.legend(loc="lower right")
      ax.set_ylabel("Avg. Attention Entropy (nats)")
      ax.set_xlabel("Layer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="Location of the pre-trained BERT model.")
    parser.add_argument("--data-file", required=True,
                        help="Name of the data file.")
    parser.add_argument("--original-corpus-file", required=True,
                        help="Name of the original corpus file.")
    parser.add_argument("--labels-file", required=False,
                        help="Name of the file containing labels.")
    args = parser.parse_args()

    ##### LOADING DATA #####

    # BERT-base Attention Maps extracted from Wikipedia
    # Data is a list of dicts of the following form:
    # {
    #    "tokens": list of strings
    #    "attns": [n_layers, n_heads, n_tokens, n_tokens]
    #             tensor of attention weights
    # }
    data = load_pickle(os.path.join(args.model_dir, "analysis/" + args.data_file + "_attn.pkl"))
    n_docs = len(data)

    # Average Jenson-Shannon divergences between attention heads
    #js_divergences = load_pickle("models/ro-trans-cased_v1/analysis/dev-source_dist.pkl")

    print('Analyzing ' + str(n_docs) + ' documents...')

    if not args.labels_file:
        labels = np.array([line.strip().split('\t')[1] for line in open(args.original_corpus_file)])
        labels = np.array(labels, dtype=int)
    else:
        labels = np.array([0 if line.strip() == 'RO' else 1 for line in open(args.labels_file)])

    # Report the number of sentences.
    print('Number of labels: {:,}\n'.format(len(labels)))

    ro_indices = np.where(labels == 0)[0]
    md_indices = np.where(labels == 1)[0]

    #### Computing Average Attention to Particular Tokens/Positions #####

    compute_token_stats = False

    if compute_token_stats:
        avg_attns = {
            k: np.zeros((12, 12)) for k in [
                "self", "right", "left", "sep", "sep_sep", "rest_sep",
                "cls", "punct"]
        }

        print("Computing token stats")
        for tokens, attns in data_iterator(data):
            n_tokens = attns.shape[-1]

            # create masks indicating where particular tokens are
            seps, clss, puncts = (np.zeros(n_tokens) for _ in range(3))
            for position, token in enumerate(tokens):
                if token == "[SEP]":
                    seps[position] = 1
                if token == "[CLS]":
                    clss[position] = 1
                if token == "." or token == ",":
                    puncts[position] = 1

            # create masks indicating which positions are relevant for each key
            sep_seps = np.ones((n_tokens, n_tokens))
            sep_seps *= seps[np.newaxis]
            sep_seps *= seps[:, np.newaxis]

            rest_seps = np.ones((n_tokens, n_tokens))
            rest_seps *= (np.ones(n_tokens) - seps)[:, np.newaxis]
            rest_seps *= seps[np.newaxis]

            selectors = {
                "self": np.eye(n_tokens, n_tokens),
                "right": np.eye(n_tokens, n_tokens, 1),
                "left": np.eye(n_tokens, n_tokens, -1),
                "sep": np.tile(seps[np.newaxis], [n_tokens, 1]),
                "sep_sep": sep_seps,
                "rest_sep": rest_seps,
                "cls": np.tile(clss[np.newaxis], [n_tokens, 1]),
                "punct": np.tile(puncts[np.newaxis], [n_tokens, 1]),
            }

            # get the average attention for each token type
            for key, selector in selectors.items():
                if key == "sep_sep":
                    denom = 2
                elif key == "rest_sep":
                    denom = n_tokens - 2
                else:
                    denom = n_tokens
                avg_attns[key] += (
                        (attns * selector[np.newaxis, np.newaxis]).sum(-1).sum(-1) /
                        (n_docs * denom))

    ##### Computing Attention Head Entropies #####

    compute_entropies = False

    if compute_entropies:
        uniform_attn_entropy = 0  # entropy of uniform attention
        entropies = np.zeros((12, 12))  # entropy of attention heads
        entropies_cls = np.zeros((12, 12))  # entropy of attention from [CLS]

        print("Computing entropy stats")
        for tokens, attns in data_iterator(data):
            attns = 0.9999 * attns + (0.0001 / attns.shape[-1])  # smooth to avoid NaNs
            uniform_attn_entropy -= np.log(1.0 / attns.shape[-1])
            entropies -= (attns * np.log(attns)).sum(-1).mean(-1)
            entropies_cls -= (attns * np.log(attns))[:, :, 0].sum(-1)

        uniform_attn_entropy /= n_docs
        entropies /= n_docs
        entropies_cls /= n_docs

    # Pretty colors
    BLACK = "k"
    GREEN = "#59d98e"
    SEA = "#159d82"
    BLUE = "#3498db"
    PURPLE = "#9b59b6"
    GREY = "#95a5a6"
    RED = "#e74c3c"
    ORANGE = "#f39c12"

    ##### Most important words based on label #####
    token_importance_plots = True

    if token_importance_plots:
        label_groups = [
            (ro_indices, 'RO'),
            (md_indices, 'MD')
        ]

        corpora = {'RO': {}, 'MD': {}}

        for indices, label_name in label_groups:
            data = np.array(data)
            for example in data[indices]:
                for token in example['tokens']:
                    if token not in corpora[label_name]:
                        corpora[label_name][token] = 0
                    corpora[label_name][token] += 1

    import pdb
    pdb.set_trace()

    if token_importance_plots:
        label_groups = [
            (ro_indices, 'RO'),
            (md_indices, 'MD')
        ]

        groups_words_attns = {}

        for indices, label_name in label_groups:
            data = np.array(data)
            target = []
            words_attns = {}
            tmp_cnt = 0
            for example in data[indices]:
                example_words_attns = get_example_attn(example, target)
                words_attns = {word: words_attns.get(word, 0) + example_words_attns.get(word, 0) for word in
                               set(words_attns) | set(example_words_attns)}
                tmp_cnt += 1

                if tmp_cnt % 500 == 0:
                    print('Done: ' + str(tmp_cnt) + '/' + str(len(data[indices])))

            words_attns = {word: attn / words_occurrences[word] for word, attn in words_attns.items()}
            groups_words_attns[label_name] = words_attns

            # excluded_words = ['[SEP]', '[CLS]']
            # for word in excluded_words:
            #     del words_attns[word]

            # words[label_name] = np.array(list(words_attns.keys()))
            # terms_attn[label_name] = list(words_attns.values())
            # sorted_pos = np.argsort(terms_attn)[::-1]
            # words = np.array(words)
            # terms_attn = np.array(terms_attn)
            #
            # max_num_terms = 20
            # sorted_pos = sorted_pos[:max_num_terms]
            #
            # plt.figure(figsize=(max_num_terms * 2.0, 15))
            # font = {'size': 16}
            # matplotlib.rc('font', **font)
            # plt.bar(words[sorted_pos], terms_attn[sorted_pos], align='center')
            # plt.xlabel('Words')
            # plt.ylabel('Average Attention')
            # plt.savefig(os.path.join(args.model_dir,
            #                          'output/token_importance_plot_' + label_name + '_labels.png'))

        for idx in range(2):
            group_words_attns = groups_words_attns[label_groups[idx][1]]
            other_words_attns = groups_words_attns[label_groups[1 - idx][1]]

            group_words = set(group_words_attns.keys())
            other_words = set(other_words_attns.keys())

            common_words = group_words.intersection(other_words)
            group_attn_diffs = {}
            for word in common_words:
                group_attn_diffs[word] = group_words_attns[word] - other_words_attns[word]

            excluded_words = ['[SEP]', '[CLS]']
            for word in excluded_words:
                del group_attn_diffs[word]

            words = np.array(list(group_attn_diffs.keys()))
            terms_attn = list(group_attn_diffs.values())
            sorted_pos = np.argsort(terms_attn)[::-1]
            words = np.array(words)
            terms_attn = np.array(terms_attn)

            max_num_terms = 8
            sorted_pos = sorted_pos[:max_num_terms]

            plt.figure(figsize=(max_num_terms * 1.2, 6))
            font = {'size': 14}
            matplotlib.rc('font', **font)
            plt.bar(words[sorted_pos], terms_attn[sorted_pos], align='center')
            plt.xlabel('Words')
            plt.ylabel('Attention Difference')
            plt.savefig(os.path.join(args.model_dir,
                                     'output/token_importance_plot_' + label_groups[idx][1] + '_labels.png'))

    ##### Avg. Attention Plots #####
    avg_attn_plots = False

    if avg_attn_plots:
        plt.figure(figsize=(5, 10))
        font = {'size': 12}
        matplotlib.rc('font', **font)
        ax = plt.subplot(3, 1, 1)
        for key, color, label in [
            ("cls", RED, "[CLS]"),
            ("sep", BLUE, "[SEP]"),
            ("punct", PURPLE, ". or ,"),
        ]:
            add_line(avg_attns, key, ax, color, label)

        ax = plt.subplot(3, 1, 2)
        for key, color, label in [
            ("rest_sep", BLUE, "other -> [SEP]"),
            ("sep_sep", GREEN, "[SEP] -> [SEP]"),
        ]:
            add_line(avg_attns, key, ax, color, label)

        ax = plt.subplot(3, 1, 3)
        for key, color, label in [
            ("left", RED, "next token"),
            ("right", BLUE, "prev token"),
            ("self", PURPLE, "current token"),
        ]:
            add_line(avg_attns, key, ax, color, label, plot_avgs=False)

        plt.savefig(os.path.join(args.model_dir,
                                 'output/avg_attn_plot.png'))

    ##### Entropy Plots #####
    entropy_plots = False

    if entropy_plots:
        xs, es, avg_es = get_data_points(entropies)
        xs, es_cls, avg_es_cls = get_data_points(entropies_cls)

        plt.figure(figsize=(5, 5))
        font = {'size': 12}
        matplotlib.rc('font', **font)

        plot_entropies(xs, uniform_attn_entropy, plt.subplot(2, 1, 1), es, avg_es, "BERT Heads",
                       c=BLUE)
        plot_entropies(xs, uniform_attn_entropy, plt.subplot(2, 1, 2), es_cls, avg_es_cls,
                       "BERT Heads from [CLS]", c=RED)

        plt.savefig(os.path.join(args.model_dir,
                                 'output/entropy_plot.png'))


if __name__ == '__main__':
    main()
