# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from Foil_Trees import domain_mappers, contrastive_explanation

# SEED = np.random.RandomState(1994)

# # Load data
# dataset_path = Path("Datasets/heart.csv")
# df = pd.read_csv(dataset_path)

# # Preprocess
# X = df.drop('target', axis=1).values
# y = df['target'].values

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=SEED
# )

# # Get names
# feature_names = df.columns.drop('target').tolist()
# target_names = ["No Heart Disease", "Heart Disease"]  # Explicit mapping

# # Domain mapper
# dm = domain_mappers.DomainMapperTabular(
#     train_data=X_train,
#     feature_names=feature_names,
#     contrast_names=target_names
# )

# # Train model
# model = RandomForestClassifier(random_state=SEED).fit(X_train, y_train)

# # Evaluation
# print('F1 Score:', metrics.f1_score(y_test, model.predict(X_test), average='weighted'))

# # Explanation
# tno = 2
# sample = X_test[tno]
# print('\nFeatures:', feature_names)
# print('Sample:', sample)
# print('\nTrue:', target_names[y_test[tno]])
# print('Predicted:', target_names[model.predict([sample])[0]])

# exp = contrastive_explanation.ContrastiveExplanation(dm)
# print("\n", exp.explain_instance_domain(model.predict_proba, sample))


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from Foil_Trees import domain_mappers, contrastive_explanation, explanators

SEED = np.random.RandomState(1994)

# Load data and preprocess
dataset_path = Path("Datasets/heart.csv")
df = pd.read_csv(dataset_path)
X = df.drop('target', axis=1).values
y = df['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Get names
feature_names = df.columns.drop('target').tolist()
target_names = ["No Heart Disease", "Heart Disease"]

# Train model
model = RandomForestClassifier(random_state=SEED).fit(X_train, y_train)

# Evaluate
print('F1 Score:', metrics.f1_score(y_test, model.predict(X_test), average='weighted'))

# Explanation setup
tno = 2
sample = X_test[tno]
true_label = y_test[tno]
predicted_label = model.predict([sample])[0]

print('\nFeatures:', feature_names)
print('Sample:', sample)
print('\nTrue:', target_names[true_label])
print('Predicted:', target_names[predicted_label])

# Existing contrastive explanation
dm = domain_mappers.DomainMapperTabular(
    train_data=X_train,
    feature_names=feature_names,
    contrast_names=target_names
)
exp = contrastive_explanation.ContrastiveExplanation(dm)
print("\nBase Explanation:", exp.explain_instance_domain(model.predict_proba, sample))

# --- Add TreeExplanator ---
# Prepare binary classification: predicted class vs others
fact = predicted_label  # Class we want to explain
foil = 1 - fact         # Contrast class
y_binary = np.where(y_train == fact, 1, 0)  # 1=fact, 0=foil

# Initialize and train tree explainer
tree_explainer = explanators.TreeExplanator(
    generalize=2,  # Control tree complexity
    verbose=True,
    seed=SEED
)

# Get decision path explanation
path, confidence, fidelity = tree_explainer.get_rule(
    fact_sample=sample,
    fact=fact,
    foil=foil,
    xs=X_train,
    ys=y_binary,
    weights=None,
    foil_strategy='informativeness'
)

# Print explanation
print("\nTree Explanation:")
if path:
    for i, rule in enumerate(path):
        feat_idx = rule[0]
        threshold = rule[1]
        value = rule[2]
        print(f"Rule {i+1}: {feature_names[feat_idx]} > {threshold:.2f}?"
              f" (Value = {value:.2f}, Met = {rule[4]})")
    print(f"\nConfidence: {confidence:.2f}, Fidelity: {fidelity:.2f}")
else:
    print("No contrastive path found")

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(
    tree_explainer.tree,
    filled=True,
    feature_names=feature_names,
    class_names=[f"Not {target_names[fact]}", target_names[fact]]
)
plt.title(f"Contrastive Tree: {target_names[fact]} vs {target_names[foil]}")
plt.show()


from sklearn.tree import export_text

tree_rules = export_text(
    tree_explainer.tree,
    feature_names=feature_names,
    decimals=2
)
print("\nDecision Tree Rules:\n", tree_rules)
