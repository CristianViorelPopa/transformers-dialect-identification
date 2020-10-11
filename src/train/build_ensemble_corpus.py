import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", required=True,
						help="Location of the data to predict on.")
	parser.add_argument("--output-dir", required=True,
						help="Location where the output corpus will be stored.")
	args = parser.parse_args()

	# Load the dataset into a pandas dataframe.
	df = pd.read_csv(args.data_file, delimiter='\t', header=None, names=['sentence', 'label'])

	# Report the number of sentences.
	print('Number of test sentences: {:,}\n'.format(df.shape[0]))

	# Create sentence and label lists
	labels = df.label.values

	pred_files = [
		'models/mbert_v5/output/dev-target/preds.npy',
		'models/ro-trans-cased_v2/output/dev-target/preds.npy',
		'models/ro-trans-uncased_v2/output/dev-target/preds.npy',
		'models/xlm-roberta_v5/output/dev-target/preds.npy',
		'models/xlm_v3/output/dev-target/preds.npy',
	]
	preds = []
	for pred_file in pred_files:
		preds.append(np.load(pred_file))
	preds = np.array(preds)
	preds = np.transpose(preds)

	X_train, X_test, y_train, y_test = train_test_split(preds, labels, test_size=0.5,
														stratify=labels)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	dump_svmlight_file(X_train, y_train, os.path.join(args.output_dir, 'train.libsvm'))
	dump_svmlight_file(X_test, y_test, os.path.join(args.output_dir, 'test.libsvm'))


if __name__ == '__main__':
	main()
