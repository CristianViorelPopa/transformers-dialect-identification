import argparse
import os
import pickle

import numpy as np
from sklearn.datasets import load_svmlight_file


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", required=True,
						help="Location of the data to predict on.")
	parser.add_argument("--model-file", required=True,
						help="Location where the model will be stored.")
	parser.add_argument("--threshold", required=True, type=float,
						help="Threshold to be used for the label prediction")
	parser.add_argument("--preds-file", required=False,
						help="Location where the predictions will be stored.")
	parser.add_argument("--pred-labels-file", required=False,
						help="Location where the predicted labels will be stored.")
	args = parser.parse_args()

	X, y = load_svmlight_file(args.data_file)
	X = X.toarray()

	model = pickle.load(open(args.model_file, 'rb'))

	predictions = model.predict_proba(X)[:, 1]

	if args.preds_file is not None:
		np.save(os.path.join(args.preds_file), predictions)

	pred_labels = ['MD' if prediction > args.threshold else 'RO' for prediction in predictions]

	if args.pred_labels_file is not None:
		open(args.pred_labels_file, 'w').write('\n'.join(pred_labels))


if __name__ == '__main__':
	main()
