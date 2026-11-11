import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from preprocesamiento import preparar_datos


def generar_explicabilidad():
    """
    Calcula la explicabilidad SHAP de la red neuronal YA ENTRENADA (no reentrena nada).

    - Explica la salida de la red (probabilidad antes del calibrador), que es lo que
      aprendio el modelo; el calibrador solo reescala esa probabilidad.
    - Guarda shap_summary.png y actualiza SOLO 'top_factores' en metricas_tecnicas.joblib,
      sin tocar las metricas de ROC-AUC, PR-AUC y Brier calculadas por calibrar_modelo.py.
    """
    X_train, X_test, _, _, _, nombres_columnas = preparar_datos()
    modelo = tf.keras.models.load_model('modelo_churn.keras')

    def predecir(x):
        return modelo(np.asarray(x, dtype='float32'), training=False).numpy().flatten()

    # Fondo: 100 clientes de entrenamiento; explicamos 200 clientes de prueba
    print("Calculando valores SHAP (puede tardar unos minutos)...")
    fondo = shap.sample(X_train, 100, random_state=42)
    X_muestra = X_test[:200]

    explicador = shap.KernelExplainer(predecir, fondo)
    valores_shap = np.asarray(explicador.shap_values(X_muestra, silent=True))
    if valores_shap.ndim == 3:  # algunas versiones de shap devuelven (n, features, 1)
        valores_shap = valores_shap[..., 0]

    importancia = np.abs(valores_shap).mean(axis=0)
    df_importancia = pd.DataFrame({
        'Variable': nombres_columnas,
        'Impacto_SHAP': importancia
    }).sort_values(by='Impacto_SHAP', ascending=False)

    print("\n--- TOP 5 FACTORES DETERMINANTES DE ABANDONO (SHAP) ---")
    print(df_importancia.head(5).to_string(index=False))

    # Grafica SHAP para la app y el informe
    plt.figure(figsize=(10, 6))
    shap.summary_plot(valores_shap, X_muestra, feature_names=nombres_columnas, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150)
    plt.close()
    print("\nGrafica guardada en 'shap_summary.png'.")

    # Actualizamos solo los factores; conservamos el resto de metricas
    try:
        metricas = joblib.load('metricas_tecnicas.joblib')
    except FileNotFoundError:
        metricas = {}
    metricas['top_factores'] = df_importancia.head(5).to_dict(orient='records')
    metricas['metodo_explicabilidad'] = 'SHAP (KernelExplainer)'
    joblib.dump(metricas, 'metricas_tecnicas.joblib')
    print("Factores guardados en 'metricas_tecnicas.joblib'.")


if __name__ == '__main__':
    generar_explicabilidad()
