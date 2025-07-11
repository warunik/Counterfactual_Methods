from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from pathlib import Path
import json

# Dataset configuration
DATASETS = {
    "heart": {
        "path": "Datasets/heart.csv",
        "class_labels": {0: "No Heart Disease", 1: "Heart Disease"},
        "feature_types": {
            "age": "numeric",
            "sex": "categorical",
            "cp": "categorical",
            "trestbps": "numeric",
            "chol": "numeric",
            "fbs": "categorical",
            "restecg": "categorical",
            "thalach": "numeric",
            "exang": "categorical",
            "oldpeak": "numeric",
            "slope": "categorical",
            "ca": "numeric",
            "thal": "categorical"
        }
    },
    "diabetes": {
        "path": "Datasets/diabetes.csv",
        "class_labels": {0: "No Diabetes", 1: "Diabetes"},
        # Add similar feature_types mapping
    },
    # Add configurations for other datasets...
}

model = OllamaLLM(model="llama3.2")

# ===== PHASE 1: Data Collection ===== 
data_collection_template = """
You are an AI assistant collecting patient data for the {dataset_name} dataset.
Your task is to collect values for all features needed for diagnosis.

**Features to collect (in this order):**
{features}

**Instructions:**
1. Ask ONE question at a time for each feature in the exact order shown above.
2. After receiving the user's answer:
   - If the feature is numeric, convert to number
   - If categorical, keep as string
3. After collecting ALL features, output EXACTLY:
   ```json
   {{"status": "complete", "patient_data": [value1, value2, ...]}}
  ```
Current Progress:

Features collected: {collected_count}/{total_features}

Next feature: {next_feature}
"""

data_collection_prompt = ChatPromptTemplate.from_template(data_collection_template)
collection_chain = data_collection_prompt | model

# ===== PHASE 2: Explanation =====
explanation_template = """
You are an AI assistant providing counterfactual explanations for {dataset_name}.
Below are the original patient data and required changes:

Original Prediction: {original_prediction} ({original_class})
New Prediction: {new_prediction} ({new_class})
Confidence: {confidence}

Required Changes:
{changes}

Patient Data:
{patient_data_str}

Task:
Explain these changes in a "what-if" scenario that a patient can understand:

Describe changes in natural language

Explain how changes affect the outcome

Provide actionable advice

Use empathetic, non-technical language
"""

explanation_prompt = ChatPromptTemplate.from_template(explanation_template)
explanation_chain = explanation_prompt | model

def collect_patient_data(dataset_config):
    features = list(dataset_config["feature_types"].keys())
    total_features = len(features)
    collected_data = []
    print(f"\n=== Collecting {dataset_config['name']} Data ===")

    for i, feature in enumerate(features):
        # Generate collection prompt
        response = collection_chain.invoke({
            "dataset_name": dataset_config["name"],
            "features": ", ".join(features),
            "collected_count": len(collected_data),
            "total_features": total_features,
            "next_feature": feature
        })
        
        # Ask question
        print(f"\n[Q{i+1}/{total_features}] {response.strip()}")
        user_input = input("Your answer: ").strip()
        
        # Convert and store
        if dataset_config["feature_types"][feature] == "numeric":
            try:
                value = float(user_input)
                collected_data.append(value)
            except ValueError:
                print(f"Invalid number! Using 0 for {feature}.")
                collected_data.append(0)
        else:
            collected_data.append(user_input)

    # Return collected data in feature order
    return collected_data
def format_counterfactual(counterfactual, dataset_config):
    """Format counterfactual results for explanation"""
    # Format changes
    changes = "\n".join([
        f"- {change['feature']}: {change['current']} → {change['new']}"
        for change in counterfactual["required_changes"]
    ])

    # Format patient data
    patient_data_str = "\n".join([
        f"- {feature}: {value}" 
        for feature, value in zip(dataset_config["feature_types"].keys(), counterfactual["patient_data"])
    ])

    # Map class labels
    class_labels = dataset_config["class_labels"]

    return {
        "original_prediction": counterfactual["original_prediction"],
        "original_class": class_labels[counterfactual["original_prediction"]],
        "new_prediction": counterfactual["new_prediction"],
        "new_class": class_labels[counterfactual["new_prediction"]],
        "confidence": counterfactual["confidence"],
        "changes": changes,
        "patient_data_str": patient_data_str,
        "dataset_name": dataset_config["name"]
    }


def main():
    print("Available datasets:", list(DATASETS.keys()))
    dataset_choice = input("Select dataset: ").strip().lower()
    if dataset_choice not in DATASETS:
        print("Invalid dataset selection!")
        return

# Load dataset config
dataset_config = DATASETS['heart']
dataset_config["name"] = 'heart'.capitalize()
dataset_path = Path(dataset_config["path"])
dataset_config["df"] = pd.read_csv(dataset_path)

# Phase 1: Data collection
patient_data = collect_patient_data(dataset_config)
print(f"\nCollected data: {patient_data}")

# Simulate counterfactual processing (in real app, call your model here)
# This would be replaced with actual counterfactual generation code
counterfactual_result = {
    "patient_data": patient_data,
    "original_prediction": 1,
    "required_changes": [
        {"feature": "chol", "current": 280, "new": 200},
        {"feature": "trestbps", "current": 150, "new": 120}
    ],
    "new_prediction": 0,
    "confidence": "High"
}

# Phase 2: Explanation
formatted_cf = format_counterfactual(counterfactual_result, dataset_config)
explanation = explanation_chain.invoke(formatted_cf)

print("\n=== Counterfactual Explanation ===")
print(explanation.strip())

if __name__ == "__main__":
    main()