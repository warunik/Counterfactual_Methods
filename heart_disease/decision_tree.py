import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from Foil_Trees import domain_mappers, contrastive_explanation
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

SEED = np.random.RandomState(1994)

# Load data
dataset_path = Path("Datasets/heart.csv")
df = pd.read_csv(dataset_path)

# Preprocess
X = df.drop('target', axis=1).values
y = df['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

# Get names
feature_names = df.columns.drop('target').tolist()
target_names = ["No Heart Disease", "Heart Disease"]  # Explicit mapping

# Domain mapper
dm = domain_mappers.DomainMapperTabular(
    train_data=X_train,
    feature_names=feature_names,
    contrast_names=target_names
)

# Train model
model = DecisionTreeClassifier(
    random_state=SEED,
    max_depth=5
).fit(X_train, y_train)

# Evaluation
print('F1 Score:', metrics.f1_score(y_test, model.predict(X_test), average='weighted'))

# Explanation
sample = X_test[2]
print('\nFeatures:', feature_names)
print('Sample:', sample)
print('\nTrue:', target_names[y_test[0]])
print('Predicted:', target_names[model.predict([sample])[0]])

exp = contrastive_explanation.ContrastiveExplanation(dm)
print("\nExplanation:", exp.explain_instance_domain(model.predict_proba, sample))

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

