# Deep Retention: Sistema de Retención Selectiva de Clientes

Proyecto de Inteligencia Artificial (UCAB) · Equipo Alfa Buena Maravilla Onda-Dinamita Escuadrón Lobo

Motor predictivo basado en una **red neuronal** que estima la probabilidad de fuga de cada cliente bancario y decide, con una regla de rentabilidad neta, **a quién conviene retener y a quién no**. Se entrega con una aplicación interactiva en Streamlit para ejecutivos de cuentas y directivos.

> **Informe ejecutivo:** [Ver el informe en Google Drive](https://drive.google.com/file/d/1ReqhfvKDMq1c_oKL4CvV6PzmXgBF9TX4/view?usp=sharing)

## Qué hace

1. **Predice** la probabilidad de abandono (0.01 a 0.99) con un Perceptrón Multicapa (MLP) entrenado sobre `Customer-Churn-Records.csv` (10,000 clientes).
2. **Calibra** esa probabilidad para que se parezca a la frecuencia real de abandono.
3. **Decide** con la fórmula de ganancia neta esperada:

   `Ganancia Neta Esperada = (Probabilidad de Fuga × Efectividad del Incentivo × Margen del Cliente) − Costo del Incentivo`

   Solo se activa un plan de retención si el resultado es **mayor que cero**.
4. **Segmenta** a los clientes y asigna la acción comercial:

   | Segmento | Regla | Decisión | Incentivo |
   |---|---|---|---|
   | A: Alto Valor / Alto Riesgo | prob. ≥ 0.65, saldo ≥ $100,000 y ganancia VIP > 0 | Retención Prioritaria | Gestor VIP y Tarifa Plana ($250) |
   | B: Medio / Riesgo Moderado | prob. ≥ 0.30 y ganancia del bono > 0 | Retención Automatizada | Bonificación Transaccional ($60) |
   | C: No Rentable | prob. ≥ 0.30 y ninguna ganancia positiva | Salida Permitida | Sin intervención ($0) |
   | D: Estable | prob. < 0.30 | Sin Acción Requerida | Mantenimiento Ordinario |

5. **Explica** las decisiones con valores SHAP (factores determinantes de abandono).

## La aplicación

Cuatro pestañas y un simulador de parámetros en la barra lateral:

- **Panel de Cartera:** seis indicadores consolidados (clientes analizados, capital en riesgo crítico, saldo a intervenir, rescate esperado, inversión, ganancia neta y ROI), distribución de clientes y balance por segmento, y matriz de decisión comercial.
- **Buscador y Evaluador de Clientes:** consulta de un cliente existente (por ID o apellido) o evaluación de un cliente nuevo ingresando sus datos.
- **Análisis del Simulador Financiero:** mapa de decisión (probabilidad vs. balance) que se actualiza con los parámetros.
- **Datos para el Informe Ejecutivo:** cifras, métricas y gráfica SHAP listas para el informe.

## Cómo ejecutarlo

### Requisitos
- **Python 3.12** (64 bits). TensorFlow puede fallar con versiones más nuevas.
- Dependencias con versión fija en `requirements.txt`.

### Instalación

```bash
# 1. Crear el entorno virtual
python -m venv venv

# 2. Activarlo
#    Windows (PowerShell):  .\venv\Scripts\Activate.ps1
#    Linux / macOS:         source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Windows:** si la ruta del proyecto es muy larga (por ejemplo, dentro de OneDrive), la instalación de TensorFlow puede quedar incompleta. En ese caso crea el entorno en una ruta corta, como `C:\venv`.

### Ejecución

Desde la carpeta del proyecto (es importante: el código usa rutas relativas):

```bash
python -m streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`. Para detenerla, presiona `Ctrl + C` en la terminal.

El repositorio ya incluye el modelo entrenado y todos los artefactos, por lo que **no es necesario reentrenar** para usar la aplicación.

## Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit (interfaz) |
| `estilos.css`, `.streamlit/config.toml` | Tema y estilo visual de la aplicación |
| `motor_financiero.py` | Canal de inferencia: probabilidad calibrada, margen, ganancia neta y segmentación |
| `preprocesamiento.py` | Escalado (StandardScaler), codificación (OneHotEncoder) y partición 80/20 |
| `entrenar_modelo.py` | Entrenamiento de la red neuronal (MLP) |
| `calibrar_modelo.py` | Calibración isotónica de la probabilidad, limitada a [0.01, 0.99] |
| `generar_explicabilidad.py` | Valores SHAP y gráfica de importancia de variables |
| `modelo_churn.keras` | Pesos del modelo entrenado |
| `preprocesador.joblib`, `nombres_columnas.joblib` | Transformador ajustado y nombres de variables |
| `calibrador.joblib` | Calibrador de probabilidades |
| `metricas_tecnicas.joblib`, `shap_summary.png` | Métricas del modelo y gráfica SHAP |
| `Customer-Churn-Records.csv` | Datos |
| `requirements.txt` | Dependencias con versiones fijas |
| `docs/Informe_Ejecutivo_Deep_Retention.pdf` | Informe ejecutivo |

## Reproducir el entrenamiento (opcional)

Solo si se desea regenerar el modelo. **Reentrenar cambia el modelo y todas las cifras del informe.** Ejecutar en este orden:

```bash
python entrenar_modelo.py          # entrena la red y guarda modelo, métricas y gráfica
python calibrar_modelo.py          # recalibra y actualiza las métricas
python generar_explicabilidad.py   # recalcula SHAP y actualiza los factores
```

`python motor_financiero.py` imprime por consola el resumen financiero de la cartera, sin interfaz.

## Arquitectura del modelo

- **Entrada:** 16 variables (10 numéricas escaladas y 3 categóricas codificadas con OneHot).
- **Capas:** Dense(64) + ReLU + BatchNormalization + Dropout(0.30) → Dense(32) + ReLU + BatchNormalization + Dropout(0.20) → Dense(1) con Sigmoide.
- **Entrenamiento:** Adam (lr 0.001), `binary_crossentropy`, ponderación de clases para el desbalance, Early Stopping sobre `val_loss` con paciencia de 12 épocas.
- **Sin fuga de datos:** se excluyen los identificadores y la variable `Complain` (correlación del 99.8% con la variable objetivo); el preprocesador se ajusta solo con los datos de entrenamiento.

## Resultados (conjunto de prueba, 20%)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.8586 |
| Precision-Recall AUC | 0.6814 |
| Brier Score (probabilidad calibrada) | 0.1030 |

Sobre la cartera de 10,000 clientes, con los parámetros por defecto del simulador, el sistema destina **$215,030** a 2,029 clientes (Segmentos A y B) para un beneficio neto esperado de **$1,445,639** (ROI 672.3%). Los factores que más pesan en el abandono, según SHAP, son el número de productos, la edad, la actividad del cliente, el país (Alemania) y el género.

## Limitaciones

- Los parámetros financieros (tasa de margen de 3.5%, comisiones de $45 y $60, costo de mantenimiento de $80 y efectividades de los incentivos) son **supuestos del equipo**, no datos observados; el ROI depende directamente de ellos.
- La calibración isotónica es escalonada: 188 clientes (1.9%) quedan en los límites 0.01 o 0.99 y comparten la misma probabilidad.
- El panel evalúa los 10,000 clientes, incluidos los usados para entrenar; las métricas técnicas se calculan solo sobre el conjunto de prueba.

## Equipo

Adrian Berbesi · Cristian De Sousa · Angel Godon · Valentina Izquierdo · Rolando Rodrigo

Universidad Católica Andrés Bello (UCAB) — Inteligencia Artificial
