import pandas as pd
import numpy as np

def split_train_validation(df, val_ratio=0.2, seed=42):
    """
    Randomly splits a DataFrame into training and validation subsets.

    Parameters:
    - df (pd.DataFrame): The full dataset to split.
    - val_ratio (float): Proportion of the data to assign to the validation set.
    - seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: Training subset.
    - pd.DataFrame: Validation subset.
    """
    validation_set = df.sample(frac=val_ratio, random_state=seed)
    training_set = df.drop(validation_set.index)

    return training_set.reset_index(drop=True), validation_set.reset_index(drop=True)
