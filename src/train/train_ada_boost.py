import argparse
import pickle

from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import AdaBoostClassifier
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
		'n_estimators': [25, 50, 75, 100],
		# 'algorithm': ('SAMME', 'SAMME.R'),
		# 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
	}
	model = AdaBoostClassifier(random_state=42)
	clf = GridSearchCV(model, parameters, n_jobs=20, cv=5, refit=True, iid=False)
	clf.fit(X, y)

	pickle.dump(clf.best_estimator_, open(args.output_file, 'wb'))


if __name__ == '__main__':
	main()
