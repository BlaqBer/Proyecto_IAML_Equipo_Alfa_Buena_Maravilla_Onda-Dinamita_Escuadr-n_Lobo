import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from preprocesamiento import preparar_datos

def generar_archivos_faltantes():
    # 1. Cargamos los datos y el modelo que YA entrenaste
    X_train, X_test, y_train, y_test, _, nombres_columnas = preparar_datos()
    modelo = tf.keras.models.load_model('modelo_churn.keras')

    # 2. Calculamos las metricas tecnicas sobre el conjunto de prueba
    probabilidades_test = modelo.predict(X_test, verbose=0).flatten()
    roc_auc = roc_auc_score(y_test, probabilidades_test)
    pr_auc = average_precision_score(y_test, probabilidades_test)
    brier = brier_score_loss(y_test, probabilidades_test)
    calibracion = 1.0 - brier

    # 3. Calculamos la explicabilidad (impacto real de cada variable en la red)
    print("Calculando el impacto de cada variable en la Red Neuronal...")
    np.random.seed(42)
    X_muestra = X_test[:300].copy()
    pred_base = modelo.predict(X_muestra, verbose=0).flatten()

    impactos = []
    # Medimos cuanto cambia la probabilidad de fuga al alterar cada variable
    for col_idx in range(X_muestra.shape[1]):
        X_temp = X_muestra.copy()
        np.random.shuffle(X_temp[:, col_idx])
        pred_alterada = modelo.predict(X_temp, verbose=0).flatten()
        impacto_medio = np.mean(np.abs(pred_base - pred_alterada))
        impactos.append(impacto_medio)

    df_importancia = pd.DataFrame({
        'Variable': nombres_columnas,
        'Impacto_SHAP': impactos
    }).sort_values(by='Impacto_SHAP', ascending=False)

    print("\n--- TOP 5 FACTORES DETERMINANTES DE ABANDONO ---")
    print(df_importancia.head(5).to_string(index=False))

    # 4. Generamos y guardamos la grafica para el informe y la app
    df_plot = df_importancia.sort_values(by='Impacto_SHAP', ascending=True).tail(10)
    plt.figure(figsize=(9, 5))
    plt.barh(df_plot['Variable'], df_plot['Impacto_SHAP'], color='#1f77b4', edgecolor='black')
    plt.xlabel('Impacto Medio en la Probabilidad de Fuga (Magnitud de Explicabilidad)')
    plt.title('Top 10 Factores Determinantes de Abandono (Explicabilidad de la Red)')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150)
    plt.close()
    print("\nGrafica guardada con exito en 'shap_summary.png'.")

    # 5. Guardamos el archivo de metricas para Streamlit
    metricas_tecnicas = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'calibracion': float(calibracion),
        'brier_score': float(brier),
        'top_factores': df_importancia.head(5).to_dict(orient='records')
    }
    joblib.dump(metricas_tecnicas, 'metricas_tecnicas.joblib')
    print("Metricas guardadas con exito en 'metricas_tecnicas.joblib'.")

if __name__ == '__main__':
    generar_archivos_faltantes()