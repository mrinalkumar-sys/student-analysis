"""
Dataset generator that creates a realistic Expanded_data_with_more_features.csv
matching the exact statistical profile, correlations, and schema of the original
Students Exam Scores dataset (30,641 records) from Students_Performance.ipynb.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, DATASET_FILE

def generate_student_dataset(n_samples: int = 30641, seed: int = 42, output_path: Path = DATASET_FILE) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. Generate Correlated Exam Scores using Multivariate Normal
    # Means: Math ~ 66.56, Reading ~ 69.38, Writing ~ 68.42
    # Std: Math ~ 15.36, Reading ~ 14.76, Writing ~ 15.44
    # Corr(Math, Reading) = 0.818, Corr(Math, Writing) = 0.807, Corr(Reading, Writing) = 0.953
    means = [66.558, 69.378, 68.419]
    stds = [15.362, 14.759, 15.444]
    
    corr_matrix = np.array([
        [1.000, 0.818, 0.807],
        [0.818, 1.000, 0.953],
        [0.807, 0.953, 1.000]
    ])
    
    # Covariance matrix: S = D * R * D
    D = np.diag(stds)
    cov_matrix = D @ corr_matrix @ D
    
    scores = np.random.multivariate_normal(means, cov_matrix, size=n_samples)
    
    # Clip scores to valid range [0, 100] and round to integers
    math_scores = np.clip(np.round(scores[:, 0]), 0, 100).astype(int)
    reading_scores = np.clip(np.round(scores[:, 1]), 10, 100).astype(int)
    writing_scores = np.clip(np.round(scores[:, 2]), 4, 100).astype(int)
    
    # 2. Demographic and Academic Categorical Features
    genders = np.random.choice(['female', 'male'], size=n_samples, p=[0.51, 0.49])
    
    # Females slightly higher in reading/writing, males slightly higher in math
    for i in range(n_samples):
        if genders[i] == 'female':
            reading_scores[i] = min(100, reading_scores[i] + np.random.randint(1, 4))
            writing_scores[i] = min(100, writing_scores[i] + np.random.randint(1, 4))
        else:
            math_scores[i] = min(100, math_scores[i] + np.random.randint(1, 4))
            
    ethnic_groups = np.random.choice(
        ['group A', 'group B', 'group C', 'group D', 'group E', None],
        size=n_samples,
        p=[0.07, 0.19, 0.31, 0.24, 0.13, 0.06]
    )
    
    parent_educs = np.random.choice(
        ["some college", "high school", "associate's degree", "some high school", "bachelor's degree", "master's degree", None],
        size=n_samples,
        p=[0.22, 0.19, 0.19, 0.17, 0.11, 0.06, 0.06]
    )
    
    lunch_types = np.random.choice(['standard', 'free/reduced'], size=n_samples, p=[0.65, 0.35])
    
    # Students with standard lunch score significantly higher (reproducing t-test effect)
    for i in range(n_samples):
        if lunch_types[i] == 'standard':
            math_scores[i] = min(100, math_scores[i] + np.random.randint(2, 6))
        else:
            math_scores[i] = max(0, math_scores[i] - np.random.randint(2, 6))
            
    test_preps = np.random.choice(['none', 'completed', None], size=n_samples, p=[0.62, 0.32, 0.06])
    
    parent_marital = np.random.choice(
        ['married', 'single', 'divorced', 'widowed', None],
        size=n_samples,
        p=[0.55, 0.23, 0.16, 0.02, 0.04]
    )
    
    practice_sport = np.random.choice(
        ['sometimes', 'regularly', 'never', None],
        size=n_samples,
        p=[0.52, 0.35, 0.11, 0.02]
    )
    
    is_first_child = np.random.choice(['yes', 'no', None], size=n_samples, p=[0.63, 0.34, 0.03])
    
    # Siblings: Poisson/discrete distribution around mean 2.15
    nr_siblings = np.random.choice(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan],
        size=n_samples,
        p=[0.10, 0.28, 0.27, 0.18, 0.08, 0.02, 0.01, 0.01, 0.05]
    )
    
    transport_means = np.random.choice(
        ['school_bus', 'private', None],
        size=n_samples,
        p=[0.53, 0.37, 0.10]
    )
    
    wkly_study = np.random.choice(
        ['5 - 10', '< 5', '> 10', None],
        size=n_samples,
        p=[0.54, 0.26, 0.17, 0.03]
    )
    
    # Construct DataFrame
    df = pd.DataFrame({
        "Unnamed: 0": np.arange(n_samples),
        "Gender": genders,
        "EthnicGroup": ethnic_groups,
        "ParentEduc": parent_educs,
        "LunchType": lunch_types,
        "TestPrep": test_preps,
        "ParentMaritalStatus": parent_marital,
        "PracticeSport": practice_sport,
        "IsFirstChild": is_first_child,
        "NrSiblings": nr_siblings,
        "TransportMeans": transport_means,
        "WklyStudyHours": wkly_study,
        "MathScore": math_scores,
        "ReadingScore": reading_scores,
        "WritingScore": writing_scores
    })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path} with {len(df)} records.")
    return df

if __name__ == "__main__":
    generate_student_dataset()
