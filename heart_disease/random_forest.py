import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Foil_Trees import domain_mappers, contrastive_explanation  # Clean import after package setup

SEED = np.random.RandomState(1994)

# Data loading and preparation
diabetes = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, 
    diabetes.target, 
    test_size=0.3, 
    random_state=SEED
)

# Domain mapper setup
dm = domain_mappers.DomainMapperTabular(
    train_data=X_train,
    feature_names=diabetes.feature_names,
    contrast_names=diabetes.target_names.tolist()
)

# Model training
model = RandomForestClassifier(random_state=SEED).fit(X_train, y_train)

# Evaluation
print('Classifier performance (F1):', metrics.f1_score(
    y_test, 
    model.predict(X_test), 
    average='weighted'
))

# Explanation generation
sample = X_test[20]
print('\nFeature names:', diabetes.feature_names)
print('Sample values:', sample)

print('\nTrue class:', diabetes.target_names[y_test[5]])
print('Predicted class:', diabetes.target_names[model.predict([sample])[0]])

# Generate explanation
exp = contrastive_explanation.ContrastiveExplanation(dm)

print("\nExplanation:", exp.explain_instance_domain(model.predict_proba, sample), "\n")


#foil_class_idx = diabetes.target_names.tolist().index('versicolor')
#print("\nExplanation:", exp.explain_instance_domain(model.predict_proba, sample, foil=foil_class_idx))

# sample = X_test[5].reshape(1, -1)  # Reshape upfront
# print('\nFeature names:', iris.feature_names)
# print('Sample values:', sample.flatten())

# print('\nTrue class:', iris.target_names[y_test[5]])
# print('Predicted class:', iris.target_names[model.predict(sample)[0]])

# try:
#     exp = contrastive_explanation.ContrastiveExplanation(dm)
#     explanation = exp.explain_instance_domain(
#         model.predict_proba, 
#         sample
#     )
#     print("\nExplanation:", explanation)
    
# except Exception as e:
#     print(f"\nExplanation Error: {str(e)}")
#     raise  # Remove this line in production
