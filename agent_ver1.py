import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from Foil_Trees import domain_mappers, contrastive_explanation
import matplotlib.pyplot as plt
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os

# ========== COMMON INITIALIZATION ==========
# Load data
dataset_path = Path("Datasets/heart.csv")
df = pd.read_csv(dataset_path)
feature_columns = df.columns.drop('target').tolist()
target_names = ["No Heart Disease", "Heart Disease"]

# Domain constraints for heart disease dataset
HEART_CONSTRAINTS = {
    'age': lambda orig, new: new >= orig - 2 and new <= orig + 5,
    'sex': lambda orig, new: orig == new,
    'cp': lambda orig, new: new >= 0 and new <= 3,
    'trestbps': lambda orig, new: new >= 90 and new <= 200,
    'chol': lambda orig, new: new >= 100 and new <= 600,
    'fbs': lambda orig, new: new >= 0 and new <= 1,
    'thalach': lambda orig, new: new >= 60 and new <= 220,
    'exang': lambda orig, new: new >= 0 and new <= 1,
    'oldpeak': lambda orig, new: new >= 0,
    'ca': lambda orig, new: new >= 0 and new <= 4,
    'thal': lambda orig, new: new >= 0 and new <= 3
}

SEED = np.random.RandomState(1994)

# ========== VECTOR STORE SETUP ==========
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_heart_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        features = ", ".join([f"{col}: {row[col]}" for col in feature_columns])
        document = Document(
            page_content=features,
            metadata={"target": row["target"], "id": str(i)}
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="heart_disease",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ========== MACHINE LEARNING MODEL SETUP ==========
# Preprocess
X = df.drop('target', axis=1).values
y = df['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

# Domain mapper
dm = domain_mappers.DomainMapperTabular(
    train_data=X_train,
    feature_names=feature_columns,
    contrast_names=target_names
)

# Train model
model = DecisionTreeClassifier(random_state=SEED, max_depth=5).fit(X_train, y_train)

# ========== FEASIBILITY MODULE ==========
class FeasibilityChecker:
    def __init__(self, domain_constraints):
        self.constraints = domain_constraints
        
    def check_single(self, original, counterfactual):
        for feature, constraint_func in self.constraints.items():
            if not constraint_func(original[feature], counterfactual[feature]):
                return False
        return True
        
    def check_feasibility(self, original, counterfactuals):
        feasible = []
        for cf in counterfactuals:
            if self.check_single(original, cf):
                feasible.append(cf)
        return feasible

def generate_counterfactuals(model, instance, desired_class, n=10, step_size=0.1):
    candidates = []
    base_pred = model.predict([instance])[0]
    
    if base_pred == desired_class:
        return [instance.copy()]
    
    importance = model.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    
    for i in range(n):
        candidate = instance.copy()
        for idx in sorted_indices[:3]:
            if np.random.rand() > 0.7:
                perturbation = np.random.choice([-1, 1]) * step_size * np.std(instance)
                candidate[idx] += perturbation
        candidates.append(candidate)
    
    return candidates

feasibility_checker = FeasibilityChecker(HEART_CONSTRAINTS)

# ========== LLM AGENT SETUP ==========
llm_model = OllamaLLM(model="llama3.2")

template = """
You are a medical AI assistant specialized in heart disease counterfactual explanations. 
Your task is to generate actionable "what-if" scenarios that would change a patient's diagnosis.

### Context:
- Dataset features: {features}
- 0 = No Heart Disease, 1 = Heart Disease

### Similar Patient Records:
{records}

### Patient Query:
{query}

### Validated Counterfactuals:
{counterfactuals}

### Task:
1. Analyze the patient's current features
2. Explain the validated counterfactual scenarios in natural language
3. Provide clinical interpretation of why these changes would work
4. Note any feasibility constraints applied
5. Output format:
   - Original Prediction: [0/1]
   - Recommended Changes: 
        • Feature: [current] → [new] (Reason: [clinical justification])
        • ...
   - Expected Outcome: [0/1]
   - Confidence: [High/Medium/Low] based on similar cases
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm_model

def format_patient_input(feature_values):
    return ", ".join([f"{col}: {val}" for col, val in zip(feature_columns, feature_values)])

def format_counterfactual(original, counterfactual):
    changes = []
    for i, name in enumerate(feature_columns):
        orig_val = original[i]
        cf_val = counterfactual[i]
        if abs(orig_val - cf_val) > 0.01:
            changes.append(f"{name}: {orig_val:.2f} → {cf_val:.2f}")
    return "\n".join(changes)

# ========== INTEGRATED AGENT LOOP ==========
while True:
    print("\n" + "="*50)
    print("Enter patient features in order (comma separated):")
    print(", ".join(feature_columns))
    
    user_input = input("\nInput (q to quit): ").strip()
    if user_input.lower() == "q":
        break
        
    try:
        # Parse and validate input
        values = [float(x.strip()) for x in user_input.split(",")]
        if len(values) != len(feature_columns):
            raise ValueError
        
        # Create query string
        query_str = format_patient_input(values)
        print(f"\nPatient Record: {query_str}")
        
        # Convert to array for model processing
        sample = np.array(values)
        
        # Retrieve similar cases
        records = retriever.invoke(query_str)
        formatted_records = "\n".join([
            f"Record {i+1}: {r.page_content} | Target: {r.metadata['target']}" 
            for i, r in enumerate(records)
        ])
        
        # ========== STRUCTURED COUNTERFACTUAL GENERATION ==========
        original_prediction = model.predict([sample])[0]
        desired_class = 1 - original_prediction
        
        # Generate candidate counterfactuals
        candidates = generate_counterfactuals(model, sample, desired_class, n=50)
        
        # Filter by feasibility
        feasible_candidates = feasibility_checker.check_feasibility(
            dict(zip(feature_columns, sample)),
            [dict(zip(feature_columns, cand)) for cand in candidates]
        )
        
        # Convert back to array format
        feasible_cf_arrays = [np.array([cf[feature] for feature in feature_columns]) 
                            for cf in feasible_candidates]
        
        # Filter candidates that flip prediction
        valid_candidates = []
        for cf in feasible_cf_arrays:
            if model.predict([cf])[0] == desired_class:
                valid_candidates.append(cf)
        
        print(f"\nGenerated {len(candidates)} counterfactual candidates")
        print(f"Found {len(valid_candidates)} valid & feasible counterfactuals")
        
        # Format counterfactuals for LLM
        counterfactual_str = ""
        if valid_candidates:
            # Format all valid counterfactuals
            counterfactual_str = "Valid Counterfactual Scenarios:\n"
            for i, cf in enumerate(valid_candidates):
                counterfactual_str += f"\nOption {i+1}:\n"
                counterfactual_str += format_counterfactual(sample, cf)
                counterfactual_str += f"\nPrediction: {target_names[desired_class]}\n"
        else:
            counterfactual_str = "No valid counterfactuals found that are medically feasible and flip the prediction"
        
        # ========== LLM ENHANCED EXPLANATION ==========
        result = chain.invoke({
            "features": ", ".join(feature_columns),
            "records": formatted_records,
            "query": query_str,
            "counterfactuals": counterfactual_str
        })
        
        print("\nIntegrated Counterfactual Explanation:")
        print(result)
        
        # ========== STRUCTURED OUTPUT ==========
        if valid_candidates:
            # Find minimal-change counterfactual
            changes = [np.linalg.norm(cf - sample) for cf in valid_candidates]
            min_idx = np.argmin(changes)
            best_cf = valid_candidates[min_idx]
            
            print("\n\nBest Counterfactual (Validated):")
            print("Feature\t\tOriginal\tCounterfactual\tChange")
            for i, name in enumerate(feature_columns):
                orig_val = sample[i]
                cf_val = best_cf[i]
                if abs(orig_val - cf_val) > 0.01:
                    print(f"{name:15}{orig_val:.2f}\t\t{cf_val:.2f}\t\t{cf_val-orig_val:+.2f}")
            
            print("\nOriginal Prediction:", target_names[original_prediction])
            print("Counterfactual Prediction:", target_names[model.predict([best_cf])[0]])
            
            # FOIL Trees explanation
            exp = contrastive_explanation.ContrastiveExplanation(dm)
            print("\nFOIL Trees Explanation:")
            print(exp.explain_instance_domain(model.predict_proba, sample))
            
            # Visualization
            plt.figure(figsize=(20, 10))
            plot_tree(model, feature_names=feature_columns, filled=True, class_names=target_names)
            plt.title("Decision Tree Visualization")
            plt.show()
        
    except ValueError:
        print(f"ERROR: Please enter exactly {len(feature_columns)} comma-separated numerical values")
    except Exception as e:
        print(f"Error: {str(e)}")