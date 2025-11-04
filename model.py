import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

wine_dataset = pd.read_csv('winequality-red.csv', sep=';')
X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda y: 1 if y >= 7 else 0)

# Train model
model = RandomForestClassifier(class_weight='balanced')
model.fit(X, Y)

# Save the model
with open('wine_quality_model.pkl', 'wb') as file:
    pickle.dump(model, file)
