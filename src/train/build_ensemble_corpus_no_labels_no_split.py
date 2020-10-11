import argparse
import os

import numpy as np
from sklearn.datasets import dump_svmlight_file


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", required=True,
						help="Location where the output corpus will be stored.")
	args = parser.parse_args()

	pred_files = [
		# 'models/mbert_v5/output/dev-target/preds.npy',
		# 'models/ro-trans-cased_v2/output/dev-target/preds.npy',
		# 'models/ro-trans-uncased_v2/output/dev-target/preds.npy',
		# 'models/xlm-roberta_v5/output/dev-target/preds.npy',
		# 'models/xlm_v3/output/dev-target/preds.npy',

		'test-results/mbert_v5/preds.npy',
		'test-results/ro-trans-cased_v2/preds.npy',
		'test-results/ro-trans-uncased_v2/preds.npy',
		'test-results/xlm-roberta_v5/preds.npy',
		'test-results/xlm_v3/preds.npy',
	]
	preds = []
	for pred_file in pred_files:
		preds.append(np.load(pred_file))
	preds = np.array(preds)
	preds = np.transpose(preds)

	labels = np.array([0] * len(preds))

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	dump_svmlight_file(preds, labels, os.path.join(args.output_dir, 'corpus.libsvm'))


if __name__ == '__main__':
	main()
