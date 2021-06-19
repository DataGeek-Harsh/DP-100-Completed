# Import libraries
from azureml.core import Run, Model
import argparse
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--training-folder", type=str, dest='training_folder', help='training data folder')
args = parser.parse_args()
training_folder = args.training_folder

# Get the experiment run context
run = Run.get_context()

# load the prepared data file in the training folder
print("Loading Data...")
file_path = os.path.join(training_folder,'data.csv')
diabetes = pd.read_csv(file_path)

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train adecision tree model
print('Training a decision tree model...')
model = DecisionTreeClassifier().fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
fig = plt.figure(figsize=(6, 4))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
run.log_image(name = "ROC", plot = fig)
plt.show()

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'diabetes_model.pkl')
joblib.dump(value=model, filename=model_file)

# Register the model
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'diabetes_model',
               tags={'Training context':'Pipeline'},
               properties={'AUC': np.float(auc), 'Accuracy': np.float(acc)})


run.complete()
