import re
import numpy as np
from spacy.lang.ro import Romanian

import constants


nlp = Romanian()
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def preprocess_corpus(docs, labels):
    remaining_indices = [idx for idx, doc in enumerate(docs) if len(doc) >= constants.MIN_DOC_LEN]
    docs = np.array(docs)[remaining_indices]
    labels = np.array(labels)[remaining_indices]

    new_docs = []
    new_labels = []
    for doc, label in zip(docs, labels):
        tokens = re.split('(\.|!|\?)', doc)
        tokens = list(filter(len, tokens))

        # merge real sentences with trailing punctuation that the split was done by
        new_docs.append(tokens[0].strip())
        new_labels.append(label)
        for token in tokens[1:]:
            # trailing punctuation
            if len(token.strip()) <= 1:
                new_docs[-1] += token.strip()
            # keep parentheses and quotes in the same paragraph
            elif new_docs[-1].count('"') % 2 == 1 or \
                    new_docs[-1].count('(') > new_docs[-1].count(')') or \
                    new_docs[-1].count('[') > new_docs[-1].count(']') or \
                    new_docs[-1].count('{') > new_docs[-1].count('}'):
                new_docs[-1] += ' ' + token.strip()
            # real new sentence
            else:
                new_docs.append(token.strip())
                new_labels.append(label)


def preprocess_corpus_spacy(docs, labels):
    remaining_indices = [idx for idx, doc in enumerate(docs) if len(doc) >= constants.MIN_DOC_LEN]
    docs = np.array(docs)[remaining_indices]
    labels = np.array(labels)[remaining_indices]

    new_docs = []
    new_labels = []

    for doc, label in zip(docs, labels):
        sents = nlp(str(doc.replace('[MASK]', '$NE$'))).sents
        sentences = [sent.string.strip().replace('$NE$', '[MASK]') for sent in sents]
        new_docs.extend(sentences)
        new_labels.extend([label] * len(sentences))

    return new_docs, new_labels
