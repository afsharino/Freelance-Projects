# Scientific 
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as  plt
import seaborn as sns


# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Deep Learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Others
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel(r'../dataset/movie-dataset.xlsx')

# lowercase all feature names
data.rename(str.lower, axis='columns', inplace=True)

# remove NaN values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

#X = data.drop(['movie', 'gross'], axis=1)
X = data[['budget', 'sequel', 'screens']]
y = data.gross

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

dt_model = DecisionTreeRegressor(max_depth=2)
_ = dt_model.fit(X_train_scaled, y_train_scaled)

y_pred_dt= dt_model.predict(X_test_scaled)

dt_mse = mean_squared_error(y_test_scaled, y_pred_dt)
print(f"The mse of decission tree is: {dt_mse}")

dt_rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_dt))
print(f"The rmse of decission tree is: {dt_mse}")

dt_mae = mean_absolute_error(y_test_scaled, y_pred_dt)
print(f"The mae of decission tree is: {dt_mae}")

nn_model = Sequential()
nn_model.add((Dense(128, input_dim=X_train_scaled.shape[1], activation='relu')))
nn_model.add((Dense(64, activation='relu')))
nn_model.add((Dense(32, activation='relu')))

nn_model.add((Dense(1, activation='linear')))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

nn_model.compile(loss='mean_squared_error', optimizer=optimizer)
nn_model.summary()
history = nn_model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'y', label='training-loss')
plt.plot(epochs, val_loss, 'r', label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.tight_layout()
plt.show()

y_pred_nn = nn_model.predict(X_test_scaled)

nn_mse = mean_squared_error(y_test_scaled, y_pred_nn)
print(f"The mse of neural net is: {nn_mse}")

nn_rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_nn))
print(f"The rmse of neural net is: {nn_rmse}")

nn_mae = mean_absolute_error(y_test_scaled, y_pred_nn)
print(f"The mae of neural net is: {nn_mae}")
