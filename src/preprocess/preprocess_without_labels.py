import sys

from doc_preprocessing import preprocess_doc


def main():
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <corpus file> <output file>")
        exit(0)

    docs = [line.strip() for line in open(sys.argv[1])]
    print('Done loading corpus.')

    docs = list(map(preprocess_doc, docs))
    print('Done pre-processing documents.')

    # docs, labels = preprocess_corpus_spacy(docs, labels)
    print('Done pre-processing corpus.')

    # lens = [len(doc) for doc in docs]
    # plt.hist(lens, 10000)
    # plt.show()

    open(sys.argv[2], 'w').write('\n'.join(docs))


if __name__ == '__main__':
    main()
