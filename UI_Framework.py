import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from Foil_Trees import domain_mappers, contrastive_explanation
from sklearn.model_selection import train_test_split

# Load model and data (one-time setup)
SEED = np.random.RandomState(1994)
dataset_path = Path("Datasets/heart.csv")
df = pd.read_csv(dataset_path)
feature_names = df.columns.drop('target').tolist()

# Initialize model and domain mapper once
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1).values,
    df['target'].values,
    test_size=0.3,
    random_state=SEED
)

dm = domain_mappers.DomainMapperTabular(
    train_data=X_train,
    feature_names=feature_names,
    contrast_names=["No Heart Disease", "Heart Disease"]
)

model = RandomForestClassifier(random_state=SEED).fit(X_train, y_train)
exp = contrastive_explanation.ContrastiveExplanation(dm)

def get_explanation():
    try:
        # Get values from UI
        input_values = [float(entries[feature].get()) for feature in feature_names]
        
        # Convert to numpy array and reshape for prediction
        sample = np.array(input_values).reshape(1, -1)
        
        # Generate explanation
        explanation = exp.explain_instance_domain(model.predict_proba, sample)
        
        # Show results
        prediction = model.predict(sample)[0]
        result_text = (f"Prediction: {'Heart Disease' if prediction else 'No Heart Disease'}\n"
                      f"Explanation:\n{explanation[0]}")
        messagebox.showinfo("Results", result_text)
        
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for all features")

# Create main window
root = tk.Tk()
root.title("Heart Disease Predictor")
root.geometry("400x600")

# Create input fields
entries = {}
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill='both', expand=True)

for idx, feature in enumerate(feature_names):
    ttk.Label(main_frame, text=f"{feature}:").grid(row=idx, column=0, sticky='w')
    entries[feature] = ttk.Entry(main_frame)
    entries[feature].grid(row=idx, column=1, pady=2, sticky='ew')

# Explanation button
btn_frame = ttk.Frame(main_frame)
btn_frame.grid(row=len(feature_names)+1, column=0, columnspan=2, pady=10)
ttk.Button(btn_frame, text="Get Explanation", command=get_explanation).pack()

# Configure grid weights
main_frame.columnconfigure(1, weight=1)

root.mainloop()