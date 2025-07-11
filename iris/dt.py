import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from Foil_Trees import domain_mappers, contrastive_explanation
import re

SEED = np.random.RandomState(1994)

# Data loading and preparation
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=SEED)

# Domain mapper setup
dm = domain_mappers.DomainMapperTabular(
    train_data=X_train,
    feature_names=iris.feature_names,
    contrast_names=iris.target_names.tolist())

# Model training
model = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)

print('Classifier performance (F1):', metrics.f1_score(
    y_test, 
    model.predict(X_test), 
    average='weighted'
))

# Generate explanation object
exp = contrastive_explanation.ContrastiveExplanation(dm)

def parse_counterfactual_conditions(explanation_text):
    """
    Parse the counterfactual conditions from the explanation text.
    Returns a list of tuples: (feature_name, operator, threshold_value)
    """
    conditions = []
    
    # Split the explanation into lines and find the counterfactuals line
    lines = explanation_text.strip().split('\n')
    counterfactual_line = None
    
    for line in lines:
        if 'Counterfactuals' in line:
            # Extract the part after the '|' separator
            parts = line.split('|')
            if len(parts) >= 3:
                counterfactual_line = parts[2].strip()
                break
    
    if not counterfactual_line:
        return conditions
    
    # Parse conditions like "petal width (cm) > 0.798" or "sepal length (cm) <= 5.45"
    # Handle multiple conditions separated by 'and' or ','
    condition_parts = re.split(r'\s+and\s+|,\s*', counterfactual_line)
    
    for part in condition_parts:
        part = part.strip()
        if not part:
            continue
            
        # Match pattern: feature_name operator value
        match = re.match(r'(.+?)\s*([<>=!]+)\s*([0-9.]+)', part)
        if match:
            feature_name = match.group(1).strip()
            operator = match.group(2).strip()
            value = float(match.group(3))
            conditions.append((feature_name, operator, value))
    
    return conditions

def apply_counterfactual_changes(sample, conditions, feature_names):
    """
    Apply counterfactual changes to a sample based on parsed conditions.
    Returns the modified sample.
    """
    modified_sample = sample.copy()
    changes_made = []
    
    for feature_name, operator, threshold in conditions:
        # Find the feature index
        if feature_name in feature_names:
            feature_idx = list(feature_names).index(feature_name)
            current_value = modified_sample[feature_idx]
            
            # Apply the counterfactual change
            if operator == '>':
                # If condition is "feature > threshold", set to slightly above threshold
                new_value = threshold + 0.01
            elif operator == '>=':
                new_value = threshold
            elif operator == '<':
                # If condition is "feature < threshold", set to slightly below threshold
                new_value = threshold - 0.01
            elif operator == '<=':
                new_value = threshold
            else:
                continue  # Skip unknown operators
            
            modified_sample[feature_idx] = new_value
            changes_made.append(f"{feature_name}: {current_value:.3f} → {new_value:.3f}")
    
    return modified_sample, changes_made

def evaluate_counterfactuals_on_dataset(X_test, y_test, model, exp, feature_names, target_names):
    """
    Evaluate counterfactuals on the entire test dataset.
    Sample workflow:
    1. original_sample = [1,1,1] -> model predicts "setosa" 
    2. CF method suggests changes -> new_sample = [1,2,1]
    3. new_sample = [1,2,1] -> run through base model -> get prediction
    4. Compare original vs CF predictions for all samples
    """
    results = []
    successful_flips = 0
    total_samples = len(X_test)
    
    # Lists to store predictions for metric calculations
    original_predictions = []
    cf_predictions = []
    true_labels = []
    
    print(f"\nEvaluating counterfactuals on {total_samples} test samples...")
    print("="*80)
    
    for i, sample in enumerate(X_test):
        try:
            # Step 1: Get original prediction from base model
            original_pred = model.predict([sample])[0]
            original_class = target_names[original_pred]
            original_predictions.append(original_pred)
            true_labels.append(y_test[i])
            
            # Step 2: Get counterfactual explanation
            explanation = exp.explain_instance_domain(model.predict_proba, sample)
            
            # Step 3: Parse counterfactual conditions
            conditions = parse_counterfactual_conditions(explanation)
            
            if not conditions:
                results.append({
                    'sample_idx': i,
                    'original_class': original_class,
                    'true_class': target_names[y_test[i]],
                    'cf_applied': False,
                    'cf_class': None,
                    'flip_successful': False,
                    'changes_made': [],
                    'explanation': explanation
                })
                # Use original prediction as CF prediction if no CF available
                cf_predictions.append(original_pred)
                continue
            
            # Step 4: Apply counterfactual changes to create new_sample
            modified_sample, changes_made = apply_counterfactual_changes(
                sample, conditions, feature_names)
            
            # Step 5: Run new_sample through BASE MODEL to get CF prediction
            cf_pred = model.predict([modified_sample])[0]
            cf_class = target_names[cf_pred]
            cf_predictions.append(cf_pred)
            
            # Step 6: Check if class was flipped from original
            flip_successful = (original_pred != cf_pred)
            if flip_successful:
                successful_flips += 1
            
            results.append({
                'sample_idx': i,
                'original_class': original_class,
                'true_class': target_names[y_test[i]],
                'cf_applied': True,
                'cf_class': cf_class,
                'flip_successful': flip_successful,
                'changes_made': changes_made,
                'explanation': explanation
            })
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_samples} samples...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            results.append({
                'sample_idx': i,
                'original_class': original_class,
                'true_class': target_names[y_test[i]],
                'cf_applied': False,
                'cf_class': None,
                'flip_successful': False,
                'changes_made': [],
                'explanation': f"Error: {str(e)}"
            })
            # Use original prediction as CF prediction if error
            cf_predictions.append(original_pred)
    
    return results, successful_flips, original_predictions, cf_predictions, true_labels

# Run evaluation on entire test dataset
evaluation_results = evaluate_counterfactuals_on_dataset(
    X_test, y_test, model, exp, iris.feature_names, iris.target_names)

# Unpack the results
results, successful_flips, original_predictions, cf_predictions, true_labels = evaluation_results

# Convert to numpy arrays for metric calculations
original_predictions = np.array(original_predictions)
cf_predictions = np.array(cf_predictions)
true_labels = np.array(true_labels)

# Calculate evaluation metrics
total_samples = len(X_test)
cf_applied_count = sum(1 for r in results if r['cf_applied'])
flip_rate = (successful_flips / cf_applied_count) * 100 if cf_applied_count > 0 else 0

print("\n" + "="*80)
print("COUNTERFACTUAL EVALUATION SUMMARY")
print("="*80)
print(f"Total test samples: {total_samples}")
print(f"Counterfactuals applied: {cf_applied_count}")
print(f"Successful class flips: {successful_flips}")
print(f"Flip success rate: {flip_rate:.1f}%")
print(f"Samples without counterfactuals: {total_samples - cf_applied_count}")

print("\n" + "="*80)
print("ORIGINAL MODEL PERFORMANCE (on test set)")
print("="*80)
print("Original samples -> Base ML Model -> Predictions")
original_accuracy = accuracy_score(true_labels, original_predictions)
original_precision = precision_score(true_labels, original_predictions, average='weighted')
original_recall = recall_score(true_labels, original_predictions, average='weighted')
original_f1 = f1_score(true_labels, original_predictions, average='weighted')

print(f"Accuracy: {original_accuracy:.4f}")
print(f"Precision: {original_precision:.4f}")
print(f"Recall: {original_recall:.4f}")
print(f"F1-Score: {original_f1:.4f}")

print("\n" + "="*80)
print("COUNTERFACTUAL MODEL PERFORMANCE")
print("="*80)
print("Example: sample=[1,1,1] (setosa) -> CF=[1,2,1] -> Base ML Model -> new prediction")
print("Modified samples (with CF changes) -> Base ML Model -> Predictions")

cf_accuracy_score = accuracy_score(true_labels, cf_predictions)
cf_precision = precision_score(true_labels, cf_predictions, average='weighted')
cf_recall = recall_score(true_labels, cf_predictions, average='weighted')
cf_f1 = f1_score(true_labels, cf_predictions, average='weighted')

print(f"Accuracy: {cf_accuracy_score:.4f}")
print(f"Precision: {cf_precision:.4f}")
print(f"Recall: {cf_recall:.4f}")
print(f"F1-Score: {cf_f1:.4f}")

print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
print(f"{'Metric':<20} {'Original':<12} {'Counterfactual':<15} {'Difference':<12}")
print("-" * 60)
print(f"{'Accuracy':<20} {original_accuracy:<12.4f} {cf_accuracy_score:<15.4f} {cf_accuracy_score-original_accuracy:<12.4f}")
print(f"{'Precision':<20} {original_precision:<12.4f} {cf_precision:<15.4f} {cf_precision-original_precision:<12.4f}")
print(f"{'Recall':<20} {original_recall:<12.4f} {cf_recall:<15.4f} {cf_recall-original_recall:<12.4f}")
print(f"{'F1-Score':<20} {original_f1:<12.4f} {cf_f1:<15.4f} {cf_f1-original_f1:<12.4f}")

print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS")
print("="*80)

print("\nORIGINAL MODEL CLASSIFICATION REPORT:")
print("(Original samples through base model)")
print(classification_report(true_labels, original_predictions, target_names=iris.target_names))

print("\nCOUNTERFACTUAL MODEL CLASSIFICATION REPORT:")
print("(CF-modified samples through base model)")
print(classification_report(true_labels, cf_predictions, target_names=iris.target_names))

print("\nORIGINAL MODEL CONFUSION MATRIX:")
print(confusion_matrix(true_labels, original_predictions))

print("\nCOUNTERFACTUAL MODEL CONFUSION MATRIX:")
print(confusion_matrix(true_labels, cf_predictions))

# Print detailed results for first few samples
print("\nDETAILED RESULTS (First 10 samples):")
print("-" * 80)
for i, result in enumerate(results[:10]):
    print(f"\nSample {result['sample_idx'] + 1}:")
    print(f"  True class: {result['true_class']}")
    print(f"  Original prediction: {result['original_class']}")
    
    if result['cf_applied']:
        print(f"  Counterfactual prediction: {result['cf_class']}")
        print(f"  Class flip successful: {result['flip_successful']}")
        print(f"  Changes made: {', '.join(result['changes_made'])}")
        
        # Status
        if result['flip_successful']:
            print(f"  ✓ CLASS FLIPPED: {result['original_class']} → {result['cf_class']}")
        else:
            print(f"  ✗ NO CLASS CHANGE: {result['original_class']}")
    else:
        print(f"  No counterfactual conditions found")

# Create a summary DataFrame
df_results = pd.DataFrame(results)
print(f"\nSummary by original class:")
summary_stats = df_results.groupby('original_class').agg({
    'cf_applied': 'sum',
    'flip_successful': 'sum'
}).rename(columns={
    'cf_applied': 'cf_attempts', 
    'flip_successful': 'successful_flips'
})
print(summary_stats)

# Save results to CSV for further analysis
df_results.to_csv('counterfactual_evaluation_results.csv', index=False)
print(f"\nDetailed results saved to 'counterfactual_evaluation_results.csv'")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)