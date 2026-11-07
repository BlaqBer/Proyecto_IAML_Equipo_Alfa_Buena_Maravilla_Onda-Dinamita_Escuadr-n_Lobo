import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from preprocesamiento import preparar_datos

# Fijamos semillas para que siempre nos de el mismo resultado al entrenar
np.random.seed(42)
tf.random.set_seed(42)

def construir_red(num_variables):
    # Armamos el Perceptron Multicapa (MLP) siguiendo la guia del proyecto
    modelo = Sequential([
        Input(shape=(num_variables,)),
        
        # Capa oculta 1: 64 neuronas + BatchNorm + Dropout del 30%
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capa oculta 2: 32 neuronas + BatchNorm + Dropout del 20%
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Capa de salida: 1 neurona con Sigmoide para emitir probabilidad (0.00 a 1.00)
        Dense(1, activation='sigmoid')
    ])
    
    # Optimizador Adam y Binary Crossentropy (estandar para clasificacion 0 o 1)
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    return modelo

def entrenar_y_evaluar():
    # 1. Traemos los datos listos desde nuestro modulo de preprocesamiento
    X_train, X_test, y_train, y_test, pesos_clase, nombres_columnas = preparar_datos()
    
    modelo = construir_red(X_train.shape[1])
    
    # 2. Configuramos Early Stopping: frena si tras 12 epocas no mejora el val_loss
    freno_temprano = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    
    print("\nIniciando entrenamiento de la Red Neuronal...")
    historial = modelo.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=64,
        class_weight=pesos_clase,
        callbacks=[freno_temprano],
        verbose=1
    )
    
    # 3. Evaluamos el modelo en el conjunto de prueba (Test)
    probabilidades_test = modelo.predict(X_test, verbose=0).flatten()
    
    roc_auc = roc_auc_score(y_test, probabilidades_test)
    pr_auc = average_precision_score(y_test, probabilidades_test)
    # Precisión de calibración (1 - Brier Score: mientras mas cercano a 1, mejor calibrado)
    brier = brier_score_loss(y_test, probabilidades_test)
    calibracion = 1.0 - brier
    
    print("\n +-+ METRICAS TECNICAS PARA EL INFORME +-+")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    print(f"Precision de Calibracion (1 - Brier): {calibracion:.4f} (Error Brier: {brier:.4f})")
    
    # Guardamos el modelo entrenado en formato nativo de Keras
    modelo.save('modelo_churn.keras')
    print("Modelo guardado exitosamente como 'modelo_churn.keras'.")
    
    # 4. Explicabilidad con SHAP (para saber que variables pesan mas en la fuga)
    print("\nCalculando valores SHAP (esto toma unos 15-30 segundos)...")
    # Usamos una muestra de 100 clientes como base y explicamos 200 casos de prueba
    fondo = shap.sample(X_train, 100, random_state=42)
    X_muestra_shap = X_test[:200]
    
    explicador = shap.KernelExplainer(lambda x: modelo.predict(x, verbose=0).flatten(), fondo)
    valores_shap = explicador.shap_values(X_muestra_shap, silent=True)
    
    # Calculamos el impacto promedio absoluto de cada variable
    importancia = np.abs(valores_shap).mean(axis=0)
    df_importancia = pd.DataFrame({
        'Variable': nombres_columnas,
        'Impacto_SHAP': importancia
    }).sort_values(by='Impacto_SHAP', ascending=False)
    
    print("\nTop 5 Factores Determinantes de Abandono (SHAP):")
    print(df_importancia.head(5).to_string(index=False))
    
    # Guardamos grafica SHAP lista para la app y el informe en PDF
    plt.figure(figsize=(10, 6))
    shap.summary_plot(valores_shap, X_muestra_shap, feature_names=nombres_columnas, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150)
    plt.close()
    print("Grafica de explicabilidad guardada en 'shap_summary.png'.")
    
    # Guardamos un resumen tecnico para que Streamlit lo muestre en pantalla a tus companeros
    metricas_tecnicas = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'calibracion': float(calibracion),
        'brier_score': float(brier),
        'top_factores': df_importancia.head(5).to_dict(orient='records')
    }
    joblib.dump(metricas_tecnicas, 'metricas_tecnicas.joblib')

if __name__ == '__main__':
    entrenar_y_evaluar()