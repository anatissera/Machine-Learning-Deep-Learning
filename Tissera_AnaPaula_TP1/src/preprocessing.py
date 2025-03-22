import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def area_units_conversion(df):
    converted_df = df.copy()
    
    sqft_to_m2 = 1 / 10.76389999 # factor de conversión de sqft a m2

    converted_df.loc[converted_df['area_units'] == 'sqft', 'area'] *= sqft_to_m2

    converted_df.drop('area_units', axis=1, inplace=True)
    
    return converted_df    


def compute_statistics(df):
    """ Calcula los estadísticos necesarios para la normalización y el escalado. """
    stats = {
        'age_mean': df['age'].mean(),
        'age_std': df['age'].std(),
        'rooms_mean': df['rooms'].mean(),
        'rooms_std': df['rooms'].std(),
        'area_min': df['area'].min(),
        'area_max': df['area'].max(),
        'price_min': df['price'].min() if 'price' in df.columns else None,
        'price_max': df['price'].max() if 'price' in df.columns else None
    }
    return stats

def normalize_var(df, variable, stats):
    return (df[f"{variable}"] - stats[f"{variable}_mean"]) / stats[f"{variable}_std"]

def scale_df(df, stats, missing_values: bool=False):
    df_scaled = df.copy()

    if missing_values:
        df_scaled['age'] = (df_scaled['age'] - stats['age_mean']) / stats['age_std']
        df_scaled['rooms'] = (df_scaled['rooms'] - stats['rooms_mean']) / stats['rooms_std']

    else:
        # df_scaled['area'] = np.log(df_scaled['area'] + 1)
        
        # if 'price' in df_scaled.columns:
        #     df_scaled['price'] = np.log(df_scaled['price'] + 1)

        # df_scaled['area'] = (df_scaled['area'] - stats['area_min']) / (stats['area_max'] - stats['area_min'])
        
        
        df_scaled['area'] = (df_scaled['area'] - stats['area_min']) / (stats['area_max'] - stats['area_min'])

        if 'price' in df_scaled.columns:
            df_scaled['price'] = (df_scaled['price'] - stats['price_min']) / (stats['price_max'] - stats['price_min'])


        # Las variables binarias quedan igual (has_pool, is_house)

    return df_scaled

def softmax(z):
    """Función softmax para convertir logits en probabilidades."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encoding(y, num_classes):
    """Convierte la variable categórica y en one-hot encoding."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot