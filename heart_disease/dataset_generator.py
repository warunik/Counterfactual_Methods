import numpy as np
import pandas as pd

def generate_synthetic_heart_data(num_samples=1000):
    # Initialize empty dataframe
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.DataFrame(columns=columns)
    
    # Generate base values within realistic ranges
    df['age'] = np.random.randint(29, 77, num_samples)
    df['sex'] = np.random.randint(0, 2, num_samples)
    df['cp'] = np.random.randint(0, 4, num_samples)
    df['trestbps'] = np.random.randint(94, 200, num_samples)
    df['chol'] = np.random.randint(126, 564, num_samples)
    df['fbs'] = np.random.randint(0, 2, num_samples)
    df['restecg'] = np.random.randint(0, 3, num_samples)
    df['thalach'] = np.random.randint(71, 202, num_samples)
    df['exang'] = np.random.randint(0, 2, num_samples)
    df['oldpeak'] = np.round(np.random.uniform(0, 6.2, num_samples), 2)
    df['slope'] = np.random.randint(0, 3, num_samples)
    df['ca'] = np.random.randint(0, 4, num_samples)
    df['thal'] = np.random.randint(0, 3, num_samples)

    # Apply complete decision tree rules
    for idx in df.index:
        row = df.loc[idx]
        
        # Root split
        if row['cp'] <= 0.5:
            # Left branch (cp <= 0.5)
            if row['ca'] <= 0.5:
                if row['thal'] <= 2.5:
                    if row['exang'] <= 0.5:
                        if row['chol'] <= 316.5:
                            df.at[idx, 'target'] = 0
                        else:
                            if row['chol'] <= 362.0:
                                df.at[idx, 'target'] = 1
                            else:
                                df.at[idx, 'target'] = 0
                    else:
                        if row['restecg'] <= 0.5:
                            df.at[idx, 'target'] = 0
                        else:
                            if row['slope'] <= 1.5:
                                if row['trestbps'] <= 115.0:
                                    df.at[idx, 'target'] = 0
                                else:
                                    df.at[idx, 'target'] = 1
                            else:
                                df.at[idx, 'target'] = 0
                else:
                    if row['oldpeak'] <= 0.65:
                        if row['chol'] <= 237.5:
                            if row['age'] <= 42.0:
                                df.at[idx, 'target'] = 1
                            else:
                                df.at[idx, 'target'] = 0
                        else:
                            df.at[idx, 'target'] = 1
                    else:
                        df.at[idx, 'target'] = 1
            else:
                if row['trestbps'] <= 109.0:
                    if row['chol'] <= 233.5:
                        df.at[idx, 'target'] = 0
                    else:
                        df.at[idx, 'target'] = 1
                else:
                    if row['sex'] <= 0.5:
                        if row['chol'] <= 298.5:
                            df.at[idx, 'target'] = 1
                        else:
                            if row['chol'] <= 355.0:
                                df.at[idx, 'target'] = 0
                            else:
                                df.at[idx, 'target'] = 1
                    else:
                        if row['thalach'] <= 106.5:
                            if row['oldpeak'] <= 0.6:
                                df.at[idx, 'target'] = 0
                            else:
                                df.at[idx, 'target'] = 1
                        else:
                            df.at[idx, 'target'] = 1  # All thalach > 106.5 paths lead to 1
        else:
            # Right branch (cp > 0.5)
            if row['oldpeak'] <= 1.95:
                if row['age'] <= 56.5:
                    if row['chol'] <= 153.0:
                        if row['sex'] <= 0.5:
                            df.at[idx, 'target'] = 0
                        else:
                            df.at[idx, 'target'] = 1
                    else:
                        if row['trestbps'] <= 111.0:
                            if row['chol'] <= 228.0:
                                df.at[idx, 'target'] = 0
                            else:
                                if row['chol'] <= 265.5:
                                    if row['age'] <= 43.0:
                                        df.at[idx, 'target'] = 0
                                    else:
                                        df.at[idx, 'target'] = 1
                                else:
                                    df.at[idx, 'target'] = 0
                        else:
                            if row['thal'] <= 2.5:
                                df.at[idx, 'target'] = 0
                            else:
                                if row['ca'] <= 0.5:
                                    df.at[idx, 'target'] = 0
                                else:
                                    df.at[idx, 'target'] = 1
                else:
                    if row['sex'] <= 0.5:
                        if row['age'] <= 57.5:
                            df.at[idx, 'target'] = 1
                        else:
                            if row['age'] <= 59.0:
                                if row['ca'] <= 1.0:
                                    df.at[idx, 'target'] = 0
                                else:
                                    df.at[idx, 'target'] = 1
                            else:
                                if row['thalach'] <= 106.0:
                                    if row['trestbps'] <= 125.0:
                                        df.at[idx, 'target'] = 0
                                    else:
                                        df.at[idx, 'target'] = 1
                                else:
                                    df.at[idx, 'target'] = 0
                    else:
                        if row['chol'] <= 245.5:
                            if row['thalach'] <= 148.0:
                                df.at[idx, 'target'] = 0
                            else:
                                if row['ca'] <= 0.5:
                                    if row['age'] <= 65.5:
                                        df.at[idx, 'target'] = 0
                                    else:
                                        df.at[idx, 'target'] = 1
                                else:
                                    if row['thalach'] <= 168.5:
                                        df.at[idx, 'target'] = 1
                                    else:
                                        df.at[idx, 'target'] = 0
                        else:
                            if row['trestbps'] <= 119.0:
                                df.at[idx, 'target'] = 0
                            else:
                                df.at[idx, 'target'] = 1
            else:
                if row['slope'] <= 0.5:
                    df.at[idx, 'target'] = 0
                else:
                    if row['restecg'] <= 0.5:
                        if row['age'] <= 54.5:
                            df.at[idx, 'target'] = 0
                        else:
                            df.at[idx, 'target'] = 1
                    else:
                        df.at[idx, 'target'] = 1

    return df

# Generate and save data
synthetic_data = generate_synthetic_heart_data(1000)
synthetic_data.to_csv('Datasets/synthetic_heart_data.csv', index=False)

# Verify distribution
print("Class distribution:\n", synthetic_data['target'].value_counts())
print("\nSample data:")
print(synthetic_data.head(3))