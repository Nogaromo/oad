from forest import RandomForest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt


if __name__ == "__main__":
	data = pd.read_csv("sdss_redshift.csv")
	y = data['redshift']
	X = data.drop('redshift', axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train = X_train.to_numpy()
	X_test = X_test.to_numpy()
	y_train = y_train.to_numpy()
	y_test = y_test.to_numpy()
	rrf = RandomForest(max_depth=10)
	rrf.fit(X_train, y_train)
	y_pred = rrf.predict_f(X_test)
	y_pred_train = rrf.predict_f(X_train)
	data_json = {"train": np.std(y_pred_train).round(5), "test": np.std(y_pred).round(5)}
	with open('redhsift.json', 'w', encoding='utf-8') as file:
		json.dump(data_json, file, ensure_ascii=False, indent=4)
	plt.figure(figsize=(19.20, 10.80), dpi=300)
	plt.grid()
	plt.title('истинное значение — предсказание')
	plt.xlabel('предсказанное значение')
	plt.ylabel('истинное значение')
	plt.plot(y_pred, y_test, 'o')
	plt.savefig('redhift.png')
	data_2 = pd.read_csv("sdss.csv")
	X = data_2.to_numpy()
	redshift_pred = rrf.predict_f(data_2)
	data_2['redshift'] = redshift_pred
	data_2.to_csv('sdss_predict.csv')

