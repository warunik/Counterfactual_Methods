import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from Foil_Trees import domain_mappers, contrastive_explanation, explanators

SEED = np.random.RandomState(1994)

diabetes = load_diabetes()

#dataset list
# df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
# df['target'] = diabetes.target
# print("Diabetes dataset:")
# print(df.head())

rx_train, X_test, ry_train, y_test = model_selection.train_test_split(diabetes.data, 
                                                                        diabetes.target, 
                                                                        train_size=0.80, 
                                                                        random_state=SEED)

m_cv = RandomForestRegressor(random_state=SEED)
model = model_selection.GridSearchCV(m_cv, cv=5, param_grid={'n_estimators': [50, 100, 500]})
model.fit(rx_train, ry_train)

print('R² Score:', metrics.r2_score(y_test, model.predict(X_test)))


dm = contrastive_explanation.DomainMapperTabular(rx_train, 
                                             feature_names=diabetes.feature_names)

exp = contrastive_explanation.ContrastiveExplanation(dm,
                                  regression=True,
                                  explanator=contrastive_explanation.TreeExplanator(verbose=True),
                                  verbose=False)

test_num = 2
sample = X_test[test_num]

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

print("\nSample Details:")
print_feature_table(diabetes.feature_names, sample)

print('\nTrue value:', y_test[test_num])
print('Predicted value:', model.predict([sample])[0])

print(exp.explain_instance_domain(model.predict, X_test[1]))

# plt.figure(figsize=(20, 10))
# plot_tree(model.best_estimator_, feature_names=diabetes.feature_names, filled=True)
# plt.title("Decision Tree Visualization")
# plt.show()

def manual_prediction():
    print("\n" + "="*40)
    print("Manual Diabetes Progression Prediction")
    print("="*40)

    try:
        # Collect manual input for all 10 diabetes features
        manual_input = []
        for feature in diabetes.feature_names:
            value = float(input(f"Enter value for {feature}: "))
            manual_input.append(value)

        manual_sample = np.array(manual_input).reshape(1, -1)

        # Get prediction and explanation
        prediction = model.predict(manual_sample)[0]
        print("Predicted progression value:", round(prediction, 2))

        print("\nExplanation:")
        print(exp.explain_instance_domain(model.predict, manual_sample))

    except ValueError:
        print("Error: Please enter valid numerical values.")

# Manual prediction loop
while True:
    manual_prediction()
    if input("\nCheck another sample? (y/n): ").lower() != 'y':
        break
