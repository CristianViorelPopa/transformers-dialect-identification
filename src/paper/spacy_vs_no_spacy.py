import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve, auc, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


# Load a trained model and vocabulary that you have fine-tuned
tokenizer_nospacy = AutoTokenizer.from_pretrained('models/mbert_v5_nospacy', do_lower_case=False)
model_nospacy = AutoModelForSequenceClassification.from_pretrained('models/mbert_v5_nospacy')

# Load a trained model and vocabulary that you have fine-tuned
tokenizer_spacy = AutoTokenizer.from_pretrained('models/mbert_v5', do_lower_case=False)
model_spacy = AutoModelForSequenceClassification.from_pretrained('models/mbert_v5')

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

# Load the dataset into a pandas dataframe.
# df = pd.read_csv(args.data_file, delimiter='\t', header=None, names=['sentence', 'label'])
data_file = 'corpus/MOROCO-Tweets/processed/test/test.tsv'
# data_file = 'corpus/RDI-Train+Dev-VARDIAL2020/processed/dev-source.tsv'
sentences = [line.strip().split('\t')[0] for line in open(data_file)]
labels = [int(line.strip().split('\t')[1]) for line in open(data_file)]

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(len(sentences)))

# Tokenize all of the sentences and map the tokens to their word IDs.
input_ids_nospacy = []
attention_masks_nospacy = []

# For every sentence...
for sent in sentences:
	# `encode_plus` will:
	#   (1) Tokenize the sentence.
	#   (2) Prepend the `[CLS]` token to the start.
	#   (3) Append the `[SEP]` token to the end.
	#   (4) Map tokens to their IDs.
	#   (5) Pad or truncate the sentence to `max_length`
	#   (6) Create attention masks for [PAD] tokens.
	encoded_dict_nospacy = tokenizer_nospacy.encode_plus(
		sent,  # Sentence to encode.
		add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
		max_length=128,  # Pad & truncate all sentences.
		pad_to_max_length=True,
		return_attention_mask=True,  # Construct attn. masks.
		return_tensors='pt',  # Return pytorch tensors.
	)

	# Add the encoded sentence to the list.
	input_ids_nospacy.append(encoded_dict_nospacy['input_ids'])

	# And its attention mask (simply differentiates padding from non-padding).
	attention_masks_nospacy.append(encoded_dict_nospacy['attention_mask'])

# Tokenize all of the sentences and map the tokens to their word IDs.
input_ids_spacy = []
attention_masks_spacy = []

# For every sentence...
for sent in sentences:
	# `encode_plus` will:
	#   (1) Tokenize the sentence.
	#   (2) Prepend the `[CLS]` token to the start.
	#   (3) Append the `[SEP]` token to the end.
	#   (4) Map tokens to their IDs.
	#   (5) Pad or truncate the sentence to `max_length`
	#   (6) Create attention masks for [PAD] tokens.
	encoded_dict_spacy = tokenizer_spacy.encode_plus(
		sent,  # Sentence to encode.
		add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
		max_length=128,  # Pad & truncate all sentences.
		pad_to_max_length=True,
		return_attention_mask=True,  # Construct attn. masks.
		return_tensors='pt',  # Return pytorch tensors.
	)

	# Add the encoded sentence to the list.
	input_ids_spacy.append(encoded_dict_spacy['input_ids'])

	# And its attention mask (simply differentiates padding from non-padding).
	attention_masks_spacy.append(encoded_dict_spacy['attention_mask'])

# Convert the lists into tensors.
input_ids_nospacy = torch.cat(input_ids_nospacy, dim=0)
attention_masks_nospacy = torch.cat(attention_masks_nospacy, dim=0)
labels = torch.tensor(labels)

# Convert the lists into tensors.
input_ids_spacy = torch.cat(input_ids_spacy, dim=0)
attention_masks_spacy = torch.cat(attention_masks_spacy, dim=0)

# Set the batch size.
batch_size = 16

# Create the DataLoader.
prediction_data_nospacy = TensorDataset(input_ids_nospacy, attention_masks_nospacy, labels)
prediction_sampler_nospacy = SequentialSampler(prediction_data_nospacy)
prediction_dataloader_nospacy = DataLoader(prediction_data_nospacy, sampler=prediction_sampler_nospacy,
								   batch_size=batch_size)

# Create the DataLoader.
prediction_data_spacy = TensorDataset(input_ids_spacy, attention_masks_spacy, labels)
prediction_sampler_spacy = SequentialSampler(prediction_data_spacy)
prediction_dataloader_spacy = DataLoader(prediction_data_spacy, sampler=prediction_sampler_spacy,
								   batch_size=batch_size)

print('Predicting labels for {:,} test sentences...'.format(len(input_ids_nospacy)))

# Put model in evaluation mode
model_nospacy.eval()
model_spacy.eval()

# Tracking variables
predictions_nospacy = []
predictions_spacy = []
true_labels = []

# Predict
for batch in prediction_dataloader_nospacy:
	# Add batch to GPU
	batch = tuple(t.to(device) for t in batch)

	# Unpack the inputs from our dataloader
	b_input_ids, b_input_mask, b_labels = batch

	# Telling the model not to compute or store gradients, saving memory and
	# speeding up prediction
	with torch.no_grad():
		# Forward pass, calculate logit predictions
		outputs = model_nospacy(b_input_ids, token_type_ids=None,
						attention_mask=b_input_mask)

	logits = outputs[0]

	# Move logits and labels to CPU
	logits = logits.detach().cpu().numpy()
	label_ids = b_labels.to('cpu').numpy()

	# Store predictions and true labels
	predictions_nospacy.extend(logits)
	true_labels.extend(label_ids)

# Predict
for batch in prediction_dataloader_spacy:
	# Add batch to GPU
	batch = tuple(t.to(device) for t in batch)

	# Unpack the inputs from our dataloader
	b_input_ids, b_input_mask, b_labels = batch

	# Telling the model not to compute or store gradients, saving memory and
	# speeding up prediction
	with torch.no_grad():
		# Forward pass, calculate logit predictions
		outputs = model_spacy(b_input_ids, token_type_ids=None,
						attention_mask=b_input_mask)

	logits = outputs[0]

	# Move logits and labels to CPU
	logits = logits.detach().cpu().numpy()
	label_ids = b_labels.to('cpu').numpy()

	# Store predictions and true labels
	predictions_spacy.extend(logits)

# predictions = np.argmax(predictions, axis=1).flatten()
predictions_nospacy = expit(np.array(predictions_nospacy)[:, 1])
print('    DONE.')

fpr_nospacy, tpr_nospacy, threshold_nospacy = roc_curve(true_labels, predictions_nospacy)
roc_auc_nospacy = auc(fpr_nospacy, tpr_nospacy)

print(roc_auc_nospacy)

predictions_spacy = expit(np.array(predictions_spacy)[:, 1])
print('    DONE.')

# predictions_spacy = np.load('news-results/mbert_v5/preds.npy')

fpr_spacy, tpr_spacy, threshold_spacy = roc_curve(true_labels, predictions_spacy)
roc_auc_spacy = auc(fpr_spacy, tpr_spacy)

print(roc_auc_spacy)

# f1s_macro_nospacy = []
# for threshold in sorted(list(predictions_nospacy)):
# 	f1s_macro_nospacy.append(f1_score(true_labels, [1 if prediction > threshold else 0 for prediction in sorted(predictions_nospacy)], average='macro'))
f1s_macro_nospacy = [f1_score(true_labels, [1 if prediction > threshold else 0 for prediction in predictions_nospacy], average='macro') for threshold in sorted(predictions_nospacy)]

# f1s_macro_spacy = []
# for threshold in sorted(list(predictions_spacy)):
# 	f1s_macro_spacy.append(f1_score(true_labels, [1 if prediction > threshold else 0 for prediction in sorted(predictions_spacy)], average='macro'))
f1s_macro_spacy = [f1_score(true_labels, [1 if prediction > threshold else 0 for prediction in predictions_spacy], average='macro') for threshold in sorted(predictions_spacy)]

import pdb
pdb.set_trace()

plt.title('Receiver Operating Characteristic')
# plt.plot(fpr_nospacy, tpr_nospacy, 'r', label='No split (AUC = %0.4f, macro-F1 = %0.4f)' % (roc_auc_nospacy, f1_score(true_labels, [1 if prediction > sorted(predictions_nospacy)[np.argmax(f1s_macro_nospacy)] else 0 for prediction in predictions_nospacy], average='macro')))
# plt.plot(fpr_spacy, tpr_spacy, 'g', label='Split (AUC = %0.4f, macro-F1 = %0.4f)' % (roc_auc_spacy, f1_score(true_labels, [1 if prediction > sorted(predictions_spacy)[np.argmax(f1s_macro_spacy)] else 0 for prediction in predictions_spacy], average='macro')))
# plt.plot(fpr_spacy, tpr_spacy, 'g', label='Split (AUC = %0.4f, macro-F1 = %0.4f)' % (roc_auc_spacy, 0.9607))
plt.plot(fpr_nospacy, tpr_nospacy, 'r', label='No split (AUC = %0.4f, macro-F1 = %0.4f)' % (roc_auc_nospacy, f1_score(true_labels, [1 if prediction > 0.1284029 else 0 for prediction in predictions_nospacy], average='macro')))
plt.plot(fpr_spacy, tpr_spacy, 'g', label='Split (AUC = %0.4f, macro-F1 = %0.4f)' % (roc_auc_spacy, f1_score(true_labels, [1 if prediction > 0.42909095 else 0 for prediction in predictions_spacy], average='macro')))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
# plt.xlim([0, 0.1])
# plt.ylim([0.9, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# plt.savefig(os.path.join(args.model_dir, 'output/roc_plot.png'))
