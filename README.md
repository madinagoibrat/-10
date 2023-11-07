import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
#кітапхана импорті

#мәліметтер қорын оқу 
data = pd.read_csv('Kazakhstan.csv')
#қажетті мәлімет алу
X = data[["Latitude", "Longitude"]]
y = data['Inflation Rate']  
#тест моделің жасау
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#бос деректер толтыру
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
#моделдің қажетті мәліметтерің жазу
params = {
    "objective": "regression",
    "num_leaves": 5,  
    "learning_rate": 0.01,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 10  
}
#моделге енгізу
model = lgb.LGBMRegressor(**params)
#график
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')  # Изменение цвета графика на синий
plt.xlabel('Фактическое изменение населения')
plt.ylabel('Прогнозируемое изменение населения')
plt.title('Изменение населения')

plt.figure(figsize=(6, 6))
data['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])  # Изменение цветов круглой диаграммы
plt.title('Распределение полов')
plt.show()

sns.scatterplot(x='Latitude', y='Longitude', data=data)
plt.title('Распределение по координатам')
plt.show()
#моделдің жауабы
print(f'RMSE на кросс-валидации: {rmse_cv.mean()}')
print(f'Коэффициент детерминации (R-squared): {r2_score(y_test, y_pred)}')
print(f'Средняя абсолютная ошибка (MAE): {mean_absolute_error(y_test, y_pred)}')
