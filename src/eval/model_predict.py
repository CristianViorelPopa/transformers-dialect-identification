import argparse

import numpy as np
from scipy.special import expit
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cased", default=False, action='store_true',
                        help="Don't lowercase the input.")
    parser.add_argument("--model-dir", required=True,
                        help="Location of the pre-trained BERT model.")
    parser.add_argument("--data-file", required=True,
                        help="Location of the data to predict on.")
    parser.add_argument("--threshold", required=True, type=float,
                        help="Threshold to be used for the label prediction")
    parser.add_argument("--preds-file", required=False,
                        help="Location where the predictions will be stored.")
    parser.add_argument("--pred-labels-file", required=False,
                        help="Location where the predicted labels will be stored.")
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
    sentences = [line.strip().split('\t')[0] for line in open(args.data_file)]

    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(len(sentences)))
    labels = [0] * len(sentences)

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
    predictions = []

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

    # predictions = np.argmax(predictions, axis=1).flatten()
    predictions = expit(np.array(predictions)[:, 1])
    print('    DONE.')

    if args.preds_file is not None:
        np.save(args.preds_file, predictions)

    pred_labels = ['MD' if prediction > args.threshold else 'RO' for prediction in predictions]

    if args.pred_labels_file is not None:
        open(args.pred_labels_file, 'w').write('\n'.join(pred_labels))


if __name__ == '__main__':
    main()
