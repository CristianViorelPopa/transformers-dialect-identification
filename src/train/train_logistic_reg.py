import argparse
import pickle

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", required=True,
						help="Location of the data to predict on.")
	parser.add_argument("--output-file", required=True,
						help="Location where the model will be stored.")
	args = parser.parse_args()

	X, y = load_svmlight_file(args.data_file)

	parameters = {
		'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		'intercept_scaling': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		'fit_intercept': (True, False)
	}
	model = LogisticRegression(penalty='l2', solver='liblinear', tol=1e-8, max_iter=500,
							   random_state=42)
	clf = GridSearchCV(model, parameters, n_jobs=20, cv=5, refit=True, iid=False)
	clf.fit(X, y)

	pickle.dump(clf.best_estimator_, open(args.output_file, 'wb'))


if __name__ == '__main__':
	main()
