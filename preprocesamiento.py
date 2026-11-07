import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

def preparar_datos(ruta_csv='Customer-Churn-Records.csv'):
    # 1. Cargamos el dataset original
    df = pd.read_csv(ruta_csv)
    print(f"Dataset cargado con exito: {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Quitamos identificadores que no aportan valor predictivo (ruido)
    # IMPORTANTE: Eliminamos 'Complain' porque tiene 99.8% de correlacion con 'Exited'
    # (se registra al momento de la baja), dejarla causaria Data Leakage severo.
    columnas_a_borrar = ['RowNumber', 'CustomerId', 'Surname', 'Exited', 'Complain']
    
    X = df.drop(columns=columnas_a_borrar)
    y = df['Exited']

    # Definimos que columnas son numeros y cuales son texto/categorias
    cols_numericas = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
        'Satisfaction Score', 'Point Earned'
    ]
    cols_categoricas = ['Geography', 'Gender', 'Card Type']

    # 2. Separamos en entrenamiento (80%) y prueba (20%) ANTES de escalar
    # Usamos stratify=y para mantener el mismo porcentaje de fugas (20.38%) en ambos lados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Armamos el transformador: escala numeros y pasa texto a One-Hot al mismo tiempo
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cols_numericas),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cols_categoricas)
        ]
    )

    # Ajustamos el preprocesador SOLO con los datos de entrenamiento (evita contaminar el test)
    X_train_proc = preprocesador.fit_transform(X_train)
    X_test_proc = preprocesador.transform(X_test)

    # Sacamos los nombres de las columnas finales (util para las graficas de SHAP despues)
    nombres_cat = preprocesador.named_transformers_['cat'].get_feature_names_out(cols_categoricas)
    nombres_columnas = cols_numericas + list(nombres_cat)

    # 4. Calculamos los pesos de cada clase para compensar que solo el 20.38% abandona
    pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    pesos_clase = {0: pesos[0], 1: pesos[1]}

    # Guardamos el preprocesador entrenado para que la app de Streamlit lo use directo
    joblib.dump(preprocesador, 'preprocesador.joblib')
    joblib.dump(nombres_columnas, 'nombres_columnas.joblib')

    print("Preprocesamiento listo y guardado en 'preprocesador.joblib'.")
    print(f"Dimensiones de entrada para la red: {X_train_proc.shape[1]} variables.")
    print(f"Pesos calculados para el desbalance: Clase 0 (Se queda) = {pesos[0]:.2f}, Clase 1 (Se va) = {pesos[1]:.2f}")

    return X_train_proc, X_test_proc, y_train.values, y_test.values, pesos_clase, nombres_columnas

if __name__ == '__main__':
    preparar_datos()