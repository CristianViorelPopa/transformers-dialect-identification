import os

import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score

import matplotlib.pyplot as plt


def add_roc(pred_file, true_labels, name, threshold=None):
    predictions = np.load(pred_file)

    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    print(roc_auc)

    if threshold:
        f1_macro = f1_score(true_labels, [1 if prediction > threshold else 0 for prediction in predictions], average='macro')

    else:
        # Find the optimal threshold for the macro-F1 score
        f1s_macro = []
        for threshold in sorted(predictions):
            f1s_macro.append(f1_score(true_labels, [1 if prediction > threshold else 0 for prediction in predictions], average='macro'))

        f1_macro = np.max(f1s_macro)

    plt.plot(fpr, tpr, label=name + ' (AUC = %0.4f' % roc_auc + ', macro-F1 = %0.4f' % f1_macro + ')')


def main():
    output_dir = 'paper'

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    tweets = False

    if tweets:
        prefix_dir = 'tweets-results'
        labels = np.array([int(line.strip().split('\t')[1]) for line in open('corpus/MOROCO-Tweets/processed/test/test.tsv')])

        add_roc(prefix_dir + '/mbert_v5/preds.npy', labels, 'mBERT', 0.4291)
        add_roc(prefix_dir + '/xlm_v3/preds.npy', labels, 'XLM', 0.49127758)
        add_roc(prefix_dir + '/xlm-roberta_v5/preds.npy', labels, 'XLM-R', 0.22)
        add_roc(prefix_dir + '/ro-trans-cased_v2/preds.npy', labels, 'Cased Rom. BERT', 0.18180618)
        add_roc(prefix_dir + '/ro-trans-uncased_v2/preds.npy', labels, 'Uncased Rom. BERT', 0.21265115)
        add_roc('test-results/svm-ensemble_v1/preds.npy', labels, 'SVM Ensemble', 0.49)

    else:
        prefix_dir = 'news-results'
        labels = np.array([int(line.strip().split('\t')[1]) for line in open('corpus/RDI-Train+Dev-VARDIAL2020/processed/dev-source.tsv')])

        add_roc(prefix_dir + '/mbert_v5/preds.npy', labels, 'mBERT', 0.61383414)
        add_roc(prefix_dir + '/xlm_v3/preds.npy', labels, 'XLM', 0.55441576)
        add_roc(prefix_dir + '/xlm-roberta_v5/preds.npy', labels, 'XLM-R', 0.48279813)
        add_roc(prefix_dir + '/ro-trans-cased_v2/preds.npy', labels, 'Cased Rom. BERT', 0.51460177)
        add_roc(prefix_dir + '/ro-trans-uncased_v2/preds.npy', labels, 'Uncased Rom. BERT', 0.6712749)
        add_roc(prefix_dir + '/svm-ensemble_v1/preds.npy', labels, 'SVM Ensemble')

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    if tweets:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    else:
        plt.xlim([0, 0.1])
        plt.ylim([0.9, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if tweets:
        plt.savefig(os.path.join(output_dir, 'ROC tweets.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'ROC news.png'))


if __name__ == '__main__':
    main()
