import argparse
import pickle

from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
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
		'n_estimators': [10, 25, 50, 75, 100],
		'criterion': ('gini', 'entropy'),
		'max_depth': [4, 8, 12, 16, 20, 24, 28, 32],
	}
	model = RandomForestClassifier(warm_start=True, random_state=42)
	clf = GridSearchCV(model, parameters, n_jobs=20, cv=5, refit=True, iid=False)
	clf.fit(X, y)

	pickle.dump(clf.best_estimator_, open(args.output_file, 'wb'))


if __name__ == '__main__':
	main()
