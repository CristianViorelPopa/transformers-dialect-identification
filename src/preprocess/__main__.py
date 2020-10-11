import sys

from doc_preprocessing import preprocess_doc
from corpus_preprocessing import preprocess_corpus, preprocess_corpus_spacy


def main():
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <corpus file> <output file>")
        exit(0)

    docs = [line.strip().split('\t')[0] for line in open(sys.argv[1])]
    labels = [line.strip().split('\t')[1] for line in open(sys.argv[1])]
    print('Done loading corpus.')
    assert len(docs) == len(labels)

    docs = list(map(preprocess_doc, docs))
    print('Done pre-processing documents.')

    docs, labels = preprocess_corpus_spacy(docs, labels)
    print('Done pre-processing corpus.')
    assert len(docs) == len(labels)

    labels = [0 if label == 'RO' else 1 for label in labels]
    print('Done replacing labels with int values.')

    # lens = [len(doc) for doc in docs]
    # plt.hist(lens, 10000)
    # plt.show()

    open(sys.argv[2], 'w').write('\n'.join([docs[idx] + '\t' + str(labels[idx])
                                            for idx in range(len(docs))]))


if __name__ == '__main__':
    main()
