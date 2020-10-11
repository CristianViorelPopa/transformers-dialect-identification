import argparse
import pickle

from sklearn.datasets import load_svmlight_file
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", required=True,
						help="Location of the data to predict on.")
	parser.add_argument("--output-file", required=True,
						help="Location where the model will be stored.")
	args = parser.parse_args()

	X, y = load_svmlight_file(args.data_file)
	X= X.toarray()

	parameters = {
		'n_restarts_optimizer': [2, 3, 4, 5, 6, 7, 8, 9, 10],
	}

	kernel = 1.0 * RBF(1.0)
	model = GaussianProcessClassifier(kernel=kernel, warm_start=True, random_state=41)
	clf = GridSearchCV(model, parameters, n_jobs=20, cv=5, refit=True, iid=False)
	clf.fit(X, y)

	pickle.dump(clf.best_estimator_, open(args.output_file, 'wb'))


if __name__ == '__main__':
	main()
