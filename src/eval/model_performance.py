import argparse
import os

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import matplotlib.pyplot as plt
import seaborn as sn

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cased", default=False, action='store_true',
                        help="Don't lowercase the input.")
    parser.add_argument("--model-dir", required=True,
                        help="Location of the pre-trained BERT model.")
    parser.add_argument("--data-file", required=True,
                        help="Location of the data to predict on.")
    parser.add_argument("--preds-file", required=False,
                        help="Location where the predictions will be stored.")
    args = parser.parse_args()

    # Load a trained model and vocabulary that you have fine-tuned
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, do_lower_case=not args.cased)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    # Tell pytorch to run this model on the GPU.
    # model.cuda()

    # Copy the model to the GPU.
    model.to(device)

    # Load the dataset into a pandas dataframe.
    # df = pd.read_csv(args.data_file, delimiter='\t', header=None, names=['sentence', 'label'])
    #
    # # Report the number of sentences.
    # print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    #
    # # Create sentence and label lists
    # sentences = df.sentence.values
    # labels = df.label.values

    sentences = np.array([line.strip().split('\t')[0] for line in open(args.data_file)])
    labels = np.array([line.strip().split('\t')[1] for line in open(args.data_file)])
    labels = np.array(labels, dtype=int)
    print('Number of test sentences: {:,}\n'.format(len(sentences)))

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Set the batch size.
    batch_size = 8

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler,
                                       batch_size=batch_size)

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.extend(logits)
        true_labels.extend(label_ids)

    predictions = expit(np.array(predictions)[:, 1])
    print('    DONE.')

    if args.preds_file is not None:
        np.save(args.preds_file, predictions)

    fpr, tpr, threshold = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    print(roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.6f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(args.model_dir, 'output/roc_plot.png'))

    f1s = []
    f1s_macro = []
    # for threshold in np.arange(0.0, 1.0, 0.01):
    for threshold in sorted(predictions):
        f1s_macro.append(f1_score(true_labels, [1 if prediction > threshold else 0 for prediction in predictions], average='macro'))
        f1s.append(f1_score(true_labels, [1 if prediction > threshold else 0 for prediction in predictions]))

    # plt.plot(sorted(list(set(predictions))), f1s)
    # plt.show()

    # import pdb
    # pdb.set_trace()

    threshold_index = np.argmax(f1s)
    macro_threshold_index = np.argmax(f1s_macro)
    # f1_macro = f1_score(true_labels, binary_predictions, average='macro')
    f1_macro = f1s_macro[macro_threshold_index]
    # f1 = f1_score(true_labels, binary_predictions)
    f1 = f1s[threshold_index]
    print('F1 = ' + str(f1) + ' with threshold ' + str(sorted(list(set(predictions)))[threshold_index]))
    print('F1 macro = ' + str(f1_macro) + ' with threshold ' + str(sorted(list(set(predictions)))[macro_threshold_index]))

    # Plot confusion matrices for both thresholds
    binary_predictions = [1 if prediction > sorted(predictions)[threshold_index] else 0 for prediction in predictions]
    cm = confusion_matrix(true_labels, binary_predictions)
    cm = cm.astype(float)
    for i in range(len(cm)):
        cm[i] = cm[i] / sum(cm[i])
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    sn.heatmap(cm, annot=True, vmin=0.0, vmax=1.0, cmap='Blues')
    plt.savefig(os.path.join(args.model_dir, 'output/confusion_matrix_f1.png'))

    binary_predictions = [1 if prediction > sorted(list(set(predictions)))[macro_threshold_index] else 0 for prediction in predictions]
    cm = confusion_matrix(true_labels, binary_predictions)
    cm = cm.astype(float)
    for i in range(len(cm)):
        cm[i] = cm[i] / sum(cm[i])
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    sn.heatmap(cm, annot=True, vmin=0.0, vmax=1.0, cmap='Blues')
    plt.savefig(os.path.join(args.model_dir, 'output/confusion_matrix_f1_macro.png'))


if __name__ == '__main__':
    main()
