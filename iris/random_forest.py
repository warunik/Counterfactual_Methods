import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Foil_Trees import domain_mappers, contrastive_explanation

SEED = np.random.RandomState(1994)

# Data loading and preparation
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target, 
    test_size=0.2,
    random_state=SEED
)

# Domain mapper setup
dm = contrastive_explanation.DomainMapperTabular(
    train_data=X_train,
    feature_names=iris.feature_names,
    contrast_names=iris.target_names.tolist()
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
sample = X_test[1]

def print_feature_table(feature_names, sample):
    """Print feature names and values in a table format"""
    # Ensure sample is 1D array
    sample = sample.reshape(-1)
    
    # Create table header
    print("|----------------------------------------|")
    print("| {:<20} | {:<15} |".format('Feature Name', 'Value'))
    print("|----------------------------------------|")
    
    # Print each feature-value pair
    for name, value in zip(feature_names, sample):
        print("| {:<20} | {:<15.2f} |".format(name, value))

    print("|----------------------------------------|")

# Usage in your existing code:
print("\nSample Details:")
print_feature_table(iris.feature_names, sample)

print('\nTrue class:', iris.target_names[y_test[5]])
print('Predicted class:', iris.target_names[model.predict([sample])[0]], '\n')

# Generate explanation
exp = contrastive_explanation.ContrastiveExplanation(dm)

print(exp.explain_instance_domain(model.predict_proba, sample))


#foil_class_idx = iris.target_names.tolist().index('versicolor')
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

# Add this section at the end of your existing code

def manual_prediction():
    print("\n" + "="*40)
    print("Manual Iris Species Prediction")
    print("="*40)
    
    try:
        # Get manual input
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))
        
        # Create sample array
        manual_sample = np.array([
            sepal_length, 
            sepal_width, 
            petal_length, 
            petal_width
        ]).reshape(1, -1)
        
        # Get prediction and explanation
        prediction = model.predict(manual_sample)[0]
        print("\n", exp.explain_instance_domain(model.predict_proba, manual_sample), "\n")
    
        
    except ValueError:
        print("Error: Please enter valid numerical values in centimeters")

# Run the manual prediction interface
while True:
    manual_prediction()
    if input("\nCheck another sample? (y/n): ").lower() != 'y':
        break