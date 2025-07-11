import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from Foil_Trees import domain_mappers, contrastive_explanation
import matplotlib.pyplot as plt

sample_idx = 4

class FeasibilityChecker:
    def __init__(self, domain_constraints):
        self.constraints = domain_constraints
        
    def check_single(self, original, counterfactual):
        """Check feasibility of a single counterfactual against constraints"""
        for feature, constraint_func in self.constraints.items():
            if not constraint_func(original[feature], counterfactual[feature]):
                return False
        return True
        
    def check_feasibility(self, original, counterfactuals):
        """Apply domain-specific constraints to filter counterfactuals"""
        feasible = []
        for cf in counterfactuals:
            if self.check_single(original, cf):
                feasible.append(cf)
        return feasible

def get_counterfactuals(model, instance, desired_class, n=10, step_size=0.1):
    """Generate candidate counterfactuals by perturbing features"""
    candidates = []
    base_pred = model.predict([instance])[0]
    
    if base_pred == desired_class:
        return [instance.copy()]  # Already in desired class
    
    # Get feature importance from the model
    importance = model.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    
    for i in range(n):
        candidate = instance.copy()
        # Perturb most important features first
        for idx in sorted_indices[:3]:  # Top 3 features
            if np.random.rand() > 0.7:  # 70% chance to perturb each important feature
                perturbation = np.random.choice([-1, 1]) * step_size * np.std(instance)
                candidate[idx] += perturbation
        candidates.append(candidate)
    
    return candidates

# Domain constraints for heart disease dataset
HEART_CONSTRAINTS = {
    'age': lambda orig, new: new >= orig,  # Age can't decrease
    'sex': lambda orig, new: orig == new,  # Gender can't change
    'cp': lambda orig, new: new >= 0 and new <= 3,  # Chest pain type must be valid category
    'trestbps': lambda orig, new: new >= 90 and new <= 200,  # Blood pressure in reasonable range
    'chol': lambda orig, new: new >= 100 and new <= 600,  # Cholesterol in possible range
    'fbs': lambda orig, new: new >= 0 and new <= 1,  # Fasting blood sugar must be binary
    'thalach': lambda orig, new: new >= 60 and new <= 220,  # Max heart rate possible
    'exang': lambda orig, new: new >= 0 and new <= 1,  # Exercise angina must be binary
    'oldpeak': lambda orig, new: new >= 0,  # ST depression can't be negative
    'ca': lambda orig, new: new >= 0 and new <= 4,  # Number of vessels 0-4
    'thal': lambda orig, new: new >= 0 and new <= 3  # Thalassemia type valid
}

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
target_names = ["No Heart Disease", "Heart Disease"]
feature_name_to_index = {name: idx for idx, name in enumerate(feature_names)}

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
sample = X_test[sample_idx]
original_prediction = model.predict([sample])[0]
desired_class = 1 - original_prediction  # Flip the prediction

# FOIL Trees explanation
exp = contrastive_explanation.ContrastiveExplanation(dm)
print("\nFOIL Trees Explanation:")
print(exp.explain_instance_domain(model.predict_proba, sample))

print('\nFeatures:', feature_names)
print('Sample:', sample)
print('\nTrue:', target_names[y_test[sample_idx]])
print('Predicted:', target_names[original_prediction])

# Visualization
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, filled=True, class_names=target_names)
plt.title("Decision Tree Visualization")
plt.show()

# Initialize feasibility checker
feasibility_checker = FeasibilityChecker(HEART_CONSTRAINTS)

# Get counterfactuals
#candidates = exp.explain_instance_domain_new(model.predict_proba, sample)
# Wrap the single instance in a list
candidates = [[59,1,1,118,210,0,0,165,0,2.4,2,0,2]]

# Now this will work
feasible_candidates = feasibility_checker.check_feasibility(
    dict(zip(feature_names, sample)),
    [dict(zip(feature_names, cand)) for cand in candidates]
)

# Filter by feasibility
feasible_candidates = feasibility_checker.check_feasibility(
    dict(zip(feature_names, sample)),
    [dict(zip(feature_names, cand)) for cand in candidates]
)

# Convert back to array format
feasible_cf_arrays = [np.array([cf[feature] for feature in feature_names]) 
                      for cf in feasible_candidates]

# Filter candidates that actually flip the prediction
valid_candidates = []
for cf in feasible_cf_arrays:
    if model.predict([cf])[0] == desired_class:
        valid_candidates.append(cf)

print(f"\nGenerated {len(candidates)} candidates")
print(f"Found {len(feasible_candidates)} feasible candidates")
print(f"Found {len(valid_candidates)} valid counterfactuals that flip prediction")

# Get minimal change counterfactual
if valid_candidates:
    # Find counterfactual with smallest change
    changes = [np.linalg.norm(cf - sample) for cf in valid_candidates]
    min_idx = np.argmin(changes)
    best_cf = valid_candidates[min_idx]
    
    # Print minimal-change counterfactual (your existing output)
    print("\nBest Counterfactual (Minimal Change):")
    print("Feature\t\tOriginal\tCounterfactual\tChange")
    for i, name in enumerate(feature_names):
        orig_val = sample[i]
        cf_val = best_cf[i]
        if abs(orig_val - cf_val) > 0.01:
            print(f"{name:15}{orig_val:.2f}\t\t{cf_val:.2f}\t\t{cf_val-orig_val:+.2f}")
    
    print("\nOriginal Prediction:", target_names[original_prediction])
    print("Counterfactual Prediction:", target_names[model.predict([best_cf])[0]])
    
else:
    print("\nNo valid counterfactuals found that are feasible and flip the prediction")
