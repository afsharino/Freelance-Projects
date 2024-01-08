
# ______________Lung Cancer ML Classification 🫁______________

# Import Libraries

# Scientific
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# MAchine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,\
                            recall_score,\
                            accuracy_score,\
                            f1_score,\
                            confusion_matrix,\
                            ConfusionMatrixDisplay,\
                            classification_report


# Load Dataset
data = pd.read_csv(r'../data/cancer patient data sets.csv')

# Data Exploration & Preproccessing

data.head(10)

# lowercase all feature names
data.rename(str.lower, axis='columns', inplace=True)

# replace space with underscore
data.rename(columns={col: col.replace(" ", "_") for col in data.columns}, inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

print(data.columns)

# Drop redundant features
data.drop(['index', 'patient_id', ], axis=1, inplace=True)

print(data.shape)

print(data.info())

print(np.unique(data.level, return_counts=True))

sns.countplot(data=data, x='level')
plt.show()


data.replace({'level':{'Low': 1, 'Medium': 2, 'High': 3}}, inplace=True)


print(data.describe().iloc[1:, ].T.style.background_gradient(axis=1))


plt.figure(figsize=(15, 10))
sns.heatmap(data.drop('level', axis=1).corr(), cmap='YlGnBu', annot=True, linewidths = 0.2);
plt.show()

# Train Test Split
X = data.drop('level', axis=1)
y = data.level

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)


# Fit Model

# ### Logistic regression

lr_model = LogisticRegression(solver='liblinear')
_ = lr_model.fit(X_train,y_train)


y_pred_lr = lr_model.predict(X_test)


lr_acc = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy of LogisticRegression: {lr_acc}")

lr_precision = precision_score(y_test, y_pred_lr, average = 'micro')
print(f"Precision of LogisticRegression: {lr_precision}")

lr_recall = recall_score(y_test, y_pred_lr, average = 'micro')
print(f"Recall of LogisticRegression: {lr_recall}")

lr_f1 = f1_score(y_test, y_pred_lr, average = 'micro')
print(f"F1 of LogisticRegression: {lr_f1}")

print('\n__________________Classification Report__________________')
print(classification_report(y_test, y_pred_lr))

cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels=['Low', 'Medium', 'High'])
cm.plot()
plt.show()

# ### RandomForest Classifier
rf_model = RandomForestClassifier(max_depth=5, n_estimators= 6, random_state=0)
_ = rf_model.fit(X_train,y_train)


y_pred_rf = rf_model.predict(X_test)


rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of RandomForest: {rf_acc}")

rf_precision = precision_score(y_test, y_pred_rf, average = 'micro')
print(f"Precision of RandomForest: {rf_precision}")

rf_recall = recall_score(y_test, y_pred_rf, average = 'micro')
print(f"Recall of RandomForest: {rf_recall}")

rf_f1 = f1_score(y_test, y_pred_rf, average = 'micro')
print(f"F1 of RandomForest: {rf_f1}")

print('\n__________________Classification Report__________________')
print(classification_report(y_test, y_pred_rf))

cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels=['Low', 'Medium', 'High'])
cm.plot()
plt.show()


# ### MLP

nn_model = MLPClassifier(random_state=1, max_iter=300)
_ = nn_model.fit(X_train,y_train)

y_pred_nn = nn_model.predict(X_test)


nn_acc = accuracy_score(y_test, y_pred_nn)
print(f"Accuracy of Multi-Layer-Perceptron: {nn_acc}")

nn_precision = precision_score(y_test, y_pred_nn, average = 'micro')
print(f"Precision of Multi-Layer-Perceptron: {nn_precision}")

nn_recall = recall_score(y_test, y_pred_nn, average = 'micro')
print(f"Recall of Multi-Layer-Perceptron: {nn_recall}")

nn_f1 = f1_score(y_test, y_pred_nn, average = 'micro')
print(f"F1 of Multi-Layer-Perceptron: {nn_f1}")

print('\n__________________Classification Report__________________')
print(classification_report(y_test, y_pred_nn))

cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_nn), display_labels=['Low', 'Medium', 'High'])
cm.plot()
plt.show()


# ## Decision Tree

dt_model = DecisionTreeClassifier()
_ = dt_model.fit(X_train,y_train)

y_pred_dt = dt_model.predict(X_test)


dt_acc = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy of Decision Tree: {dt_acc}")

dt_precision = precision_score(y_test, y_pred_dt, average = 'micro')
print(f"Precision of Decision Tree: {dt_precision}")

dt_recall = recall_score(y_test, y_pred_dt, average = 'micro')
print(f"Recall of Decision Tree: {dt_recall}")

dt_f1 = f1_score(y_test, y_pred_dt, average = 'micro')
print(f"F1 of Decision Tree: {dt_f1}")

print('\n__________________Classification Report__________________')
print(classification_report(y_test, y_pred_dt))


cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt), display_labels=['Low', 'Medium', 'High'])
cm.plot()
plt.show()


# ### KNN

list_of_neighbors = np.arange(1,101)
accs = []

for nn in list_of_neighbors:
    knn_model = KNeighborsClassifier(n_neighbors=nn)
    _ = knn_model.fit(X_train,y_train)
    y_pred_knn = knn_model.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    print(f"{nn}, Accuracy of K-Nearest Neighbors: {knn_acc}")
    accs.append(knn_acc)

sns.lineplot(x=list_of_neighbors, y=accs)
plt.title('Accuracy vs. Number of Neighbors for KNN')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=5)
_ = knn_model.fit(X_train,y_train)

y_pred_knn = knn_model.predict(X_test)


knn_acc = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of K-Nearest Neighbors: {knn_acc}")

knn_precision = precision_score(y_test, y_pred_knn, average = 'micro')
print(f"Precision of K-Nearest Neighbors: {knn_precision}")

knn_recall = recall_score(y_test, y_pred_knn, average = 'micro')
print(f"Recall of K-Nearest Neighbors: {knn_recall}")

knn_f1 = f1_score(y_test, y_pred_knn, average = 'micro')
print(f"F1 of K-Nearest Neighbors: {knn_f1}")

print('\n__________________Classification Report__________________')
print(classification_report(y_test, y_pred_knn))

cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_knn), display_labels=['Low', 'Medium', 'High'])
cm.plot()
plt.show()

