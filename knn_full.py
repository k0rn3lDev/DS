import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/synthetic_retail_sales.csv')

numeric_cols = ['UnitPrice', 'QuantitySold', 'TotalSale', 'CustomerAge']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df['ProductCategory'] = df['ProductCategory'].fillna(df['ProductCategory'].mode()[0])

le = LabelEncoder()
df['ProductCategory'] = le.fit_transform(df['ProductCategory'])

X = df[['UnitPrice', 'QuantitySold', 'TotalSale', 'CustomerAge']]
y = df['ProductCategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)