import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

def cargar_datos(path: str, filename: str) -> pd.DataFrame:
    """
    Carga los datos crudos

    :param path: Path donde está ubicado el archivo CSV con los datos crudos
    :type path: str
    :param filename: Nombre del archivo CSV
    :type filename: str
    :returns: Los datos crudos como un archivo CSV
    :rtype: pd.DataFrame
    """

    # Cargamos el dataset
    file = os.path.join(path, filename)
    return pd.read_csv(file)


def eliminar_columnas(dataset: pd.DataFrame, columnas_eliminar: list) -> pd.DataFrame:
    """
    Elimina las columnas seleccionadas

    :param dataset: Dataframe con el dataset
    :type dataset: pd.DataFrame
    :param columnas_eliminar: Columnas a eliminar
    :type columnas_eliminar: list
    :rtype: pd.DataFrame
    """
    return dataset.drop(columns=columnas_eliminar)


def eliminar_nulos_columna(dataset: pd.DataFrame, columnas_eliminar: list) -> pd.DataFrame:
    """
    Elimina las filass con nulos en las columnas seleccionadas

    :param dataset: Dataframe con el dataset
    :type dataset: pd.DataFrame
    :param columnas_eliminar: Columnas donde eliminar todas las filas con nulos
    :type columnas_eliminar: list
    :rtype: pd.DataFrame
    """
    return dataset.dropna(subset=columnas_eliminar)


def eliminar_nulos_multiples(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina las filas con más de un nulo 

    :param dataset: Dataframe con el dataset
    :type dataset: pd.DataFrame
    :rtype: pd.DataFrame
    """
    rows_to_drop = dataset[dataset.isnull().sum(axis=1) >= 2].index
    return dataset.drop(index=rows_to_drop)


def split_dataset(dataset: pd.DataFrame, test_size: float,
                  target_column: str, n_semilla: int) -> tuple:
    """
    Genera una división del dataset en una parte de entrenamiento y otra de validación

    :param dataset: Dataframe con el dataset
    :type dataset: pd.DataFrame
    :param test_size: Proporción del set de testeo
    :type test_size: float
    :param target_column: Nombre de la columna de target para el entrenamiento
    :type target_column: str
    :param n_semilla: Número de semilla para el split
    :type n_semilla: int
    :returns: Tupla con las entradas y salidas de entrenamiento y testeo.
    :rtype: tuple
    """

    X = dataset.drop(columns=target_column)
    y = dataset[[target_column]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=n_semilla)
    
    return X_train, X_test, y_train, y_test


def imputar_variables(X_train: pd.DataFrame, X_test: pd.DataFrame, variables_para_imputar: list,
                  max_iter: int, n_semilla: int) -> tuple:
    """
    Imputa valores nulos con MICE

    :param X_train: Dataframe con el dataset de entrenamiento
    :type X_train: pd.DataFrame
    :param X_test: Dataframe con el dataset de entrenamiento
    :type X_test: pd.DataFrame
    :param variables_para_imputar: Lista de variables usadas para imputar
    :type variables_para_imputar: list
    :param max_iter: Número máximo de iteraciones
    :type max_iter: int
    :param n_semilla: Número de semilla para el split
    :type n_semilla: int
    :returns: tupla de imputador entrenado, y valores de X_train y X_test convertidos
    :rtype: IterativeImputer
    """

    # Copiar dataframes originales para no modificarlos
    X_train_imputado = X_train.copy()
    X_test_imputado = X_test.copy()

    # Entrenar imputador
    imputer = IterativeImputer(max_iter=10, random_state=42)
    array_train_imputado = imputer.fit_transform(X_train[variables_para_imputar])

    # Convertir datos imputados a dataframe
    X_train_imputado[variables_para_imputar] = pd.DataFrame(
        array_train_imputado,
        columns=variables_para_imputar,
        index=X_train.index
    )

    # Transformar datos de test
    array_test_imputado  = imputer.transform(X_test[variables_para_imputar])

    # Convertir datos imputados a dataframe
    X_test_imputado[variables_para_imputar] = pd.DataFrame(
        array_test_imputado,
        columns=variables_para_imputar,
        index=X_test.index
    )
        
    return imputer, X_train_imputado, X_test_imputado


def clasificar_burn_rate(y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
    """
    Clasifica la variable target Burn Rate en categorías

    :param y_train: Target train
    :type y_train: pd.DataFrame
    :param y_test: Target test
    :type y_test: float
    :returns: Tupla con las entradas y salidas de entrenamiento y testeo.
    :rtype: tuple
    """

    column_name = y_train.columns[0]

    # Convierte los valores a Low, Medium y High dependiendo el valor de Burn Rate
    y_train_class = pd.DataFrame(
        np.select(
            [y_train[column_name] < 0.33, y_train[column_name] < 0.66],
            ["Low", "Medium"],
            default="High"
        ),
        columns=[column_name],
        index=y_train.index
    )

    y_test_class = pd.DataFrame(
        np.select(
            [y_test[column_name] < 0.33, y_test[column_name] < 0.66],
            ["Low", "Medium"],
            default="High"
        ),
        columns=[column_name],
        index=y_test.index
    )

    return y_train_class, y_test_class


def codificar_target(y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
    """
    Codifica la variable target con OrdinalEncoder

    :param y_train: Target train
    :type y_train: pd.DataFrame
    :param y_test: Target test
    :type y_test: float
    :returns: Tupla con las entradas y salidas de entrenamiento y testeo.
    :rtype: tuple
    """
    


    # Inicializar OrdinalEncoder con categorías explícitas
    ordinal_encoder = OrdinalEncoder(categories=[["Low", "Medium", "High"]], dtype=np.int32)

    # Codificar y_train
    y_train_encoded = pd.Series(
        ordinal_encoder.fit_transform(y_train).ravel(),
        name="BurnRate_Class",
        index=y_train.index
    )

    # Codificar y_test
    y_test_encoded = pd.Series(
        ordinal_encoder.transform(y_test).ravel(),
        name="BurnRate_Class",
        index=y_test.index
    )

    return ordinal_encoder, y_train_encoded, y_test_encoded


def codificar_categoricas(X_train: pd.DataFrame, X_test: pd.DataFrame, columnas_categoricas: list) -> tuple:
    """
    Codifica las columnas categoricas con One-Hot Encoder

    :param X_train: Datos train
    :type X_train: pd.DataFrame
    :param X_test: Datos test
    :type X_test: float
    param columnas_categoricas: Listado de columnas categóricas
    :type columnas_categoricas: list
    :returns: Tupla con las entradas y salidas de entrenamiento y testeo.
    :rtype: tuple
    """

    # Extraer columnas numericas
    numeric_cols = [col for col in X_train.columns if col not in columnas_categoricas]
    X_train_num_df = X_train[numeric_cols].copy()
    X_test_num_df  = X_test[numeric_cols].copy()

    # Codificación One-Hot para categóricas 
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_train_ohe = ohe.fit_transform(X_train[columnas_categoricas])
    X_test_ohe  = ohe.transform(X_test[columnas_categoricas])
    ohe_feature_names = ohe.get_feature_names_out(columnas_categoricas)

    # Convertimos las variables codificadas a DataFrames
    X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=ohe_feature_names, index=X_train.index)
    X_test_ohe_df  = pd.DataFrame(X_test_ohe,  columns=ohe_feature_names, index=X_test.index)

    X_train_full = pd.concat([X_train_num_df, X_train_ohe_df], axis=1)
    X_test_full  = pd.concat([X_test_num_df,  X_test_ohe_df],  axis=1)

    return ohe, X_train_full, X_test_full


# Proceso de Extract, Load and Transform
n_semilla = 42
dataset = cargar_datos("data", "enriched_employee_dataset.csv")
dataset = eliminar_columnas(dataset, ['Employee ID', 'Date of Joining', 'Years in Company'])
dataset = eliminar_nulos_columna(dataset, ["Burn Rate"])
dataset = eliminar_nulos_multiples(dataset)
X_train, X_test, y_train, y_test = split_dataset(dataset, 0.2, 'Burn Rate', n_semilla)
variables_para_imputar = [
    'Designation', 'Resource Allocation', 'Mental Fatigue Score',
    'Work Hours per Week', 'Sleep Hours', 'Work-Life Balance Score',
    'Manager Support Score', 'Deadline Pressure Score',
    'Recognition Frequency'
]
imputer, X_train_imputado, X_test_imputado = imputar_variables(X_train, X_test, variables_para_imputar, 10, n_semilla)
y_train_class, y_test_class = clasificar_burn_rate(y_train, y_test)
encoder_target, y_train_encoded, y_test_encoded = codificar_target(y_train_class, y_test_class)
encoder_categoricas, X_train_codif, X_test_codif = codificar_categoricas(X_train_imputado, X_test_imputado, ["Gender", "Company Type", "WFH Setup Available"])

