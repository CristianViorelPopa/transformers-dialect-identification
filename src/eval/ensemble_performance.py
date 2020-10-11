import argparse
import os
import pickle

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", required=True,
						help="Location of the data to predict on.")
	parser.add_argument("--model-file", required=True,
						help="Location where the model will be stored.")
	parser.add_argument("--output-dir", required=True,
						help="Location of the directory to save ensemble performance files")
	args = parser.parse_args()

	X, y = load_svmlight_file(args.data_file)
	X = X.toarray()

	model = pickle.load(open(args.model_file, 'rb'))

	predictions = model.predict_proba(X)[:, 1]

	np.save(os.path.join(args.output_dir, 'preds.npy'), predictions)

	import pdb
	pdb.set_trace()

	fpr, tpr, threshold = roc_curve(y, predictions)
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
	plt.savefig(os.path.join(args.output_dir, 'roc_plot.png'))

	f1s = []
	for threshold in np.arange(0.0, 1.0, 0.01):
		f1s.append(f1_score(y, [1 if prediction > threshold else 0 for prediction in predictions], average='macro'))
		# f1s.append(f1_score(y, [1 if prediction > threshold else 0 for prediction in predictions]))

	plt.plot(np.arange(0.0, 1.0, 0.01), f1s)
	plt.show()

	import pdb
	pdb.set_trace()

	threshold = 0.49
	binary_predictions = [1 if prediction > threshold else 0 for prediction in predictions]
	f1 = f1_score(y, binary_predictions, average='macro')
	# f1 = f1_score(y, binary_predictions)
	print('F1 = ' + str(f1))

	# Plot confusion matrix
	cm = confusion_matrix(y, binary_predictions)
	cm = cm.astype(float)
	for i in range(len(cm)):
		cm[i] = cm[i] / sum(cm[i])
	plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
	sn.heatmap(cm, annot=True, vmin=0.0, vmax=1.0, cmap='Blues')
	plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))


if __name__ == '__main__':
	main()
