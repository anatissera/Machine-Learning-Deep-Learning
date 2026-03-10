
def split_train_validation(df, val_ratio=0.2, seed=42):
    """
    Divide aleatoriamente un DataFrame en subconjuntos de entrenamiento y validación.

    Parámetros:
    - df (pd.DataFrame): El conjunto de datos completo a dividir.
    - val_ratio (float): Proporción de los datos a asignar al conjunto de validación.
    - seed (int): Semilla aleatoria para reproducibilidad.

    Retorna:
    - pd.DataFrame: Subconjunto de entrenamiento.
    - pd.DataFrame: Subconjunto de validación.
    """
    validation_set = df.sample(frac=val_ratio, random_state=seed)
    training_set = df.drop(validation_set.index)

    return training_set.reset_index(drop=True), validation_set.reset_index(drop=True)
