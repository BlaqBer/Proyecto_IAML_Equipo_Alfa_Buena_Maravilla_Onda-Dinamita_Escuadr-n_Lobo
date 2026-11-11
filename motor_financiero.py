import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Cargamos los artefactos entrenados una sola vez (Canal de Inferencia)
def cargar_artefactos():
    preprocesador = joblib.load('preprocesador.joblib')
    modelo = tf.keras.models.load_model('modelo_churn.keras')
    return preprocesador, modelo

# Calibrador de probabilidades (regresion isotonica generada por calibrar_modelo.py).
# La red se entreno con class_weight='balanced' y sobreestima la fuga; el calibrador
# corrige la probabilidad para que se acerque a la frecuencia real de abandono.
# Si el archivo no existe, se usan las probabilidades crudas de la red.
_CALIBRADOR = None

def _cargar_calibrador():
    global _CALIBRADOR
    if _CALIBRADOR is None and os.path.exists('calibrador.joblib'):
        _CALIBRADOR = joblib.load('calibrador.joblib')
    return _CALIBRADOR

def calcular_margen_cliente(df):
    """
    Estima el margen anual que el cliente le genera al banco (en $)
    cruzando balances, comisiones por productos y volumen transaccional (actividad),
    menos un costo operativo base para detectar cuentas deficitarias.
    """
    tasa_margen_saldo = 0.035  # El banco gana 3.5% anual sobre el balance depositado
    comision_producto = 45.0   # Ingreso anual por cada producto bancario contratado
    bono_actividad = 60.0      # Ingreso por volumen transaccional si es miembro activo
    costo_mantenimiento = 80.0 # Costo operativo anual para el banco por mantener la cuenta

    margen = (
        (df['Balance'] * tasa_margen_saldo) +
        (df['NumOfProducts'] * comision_producto) +
        (df['IsActiveMember'] * bono_actividad) -
        costo_mantenimiento
    )
    return margen

COLUMNAS_ENTRADA = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Satisfaction Score', 'Card Type', 'Point Earned'
]

def predecir_probabilidades(df, preprocesador, modelo):
    """
    Probabilidad de fuga (ya calibrada) de cada cliente. Depende solo de los datos del
    cliente y no de los parametros del simulador, por lo que la app puede guardarla en
    cache y evitar volver a ejecutar la red al mover un slider.
    """
    X_proc = preprocesador.transform(df[COLUMNAS_ENTRADA])
    prob_cruda = modelo.predict(X_proc, verbose=0).flatten()
    calibrador = _cargar_calibrador()
    return calibrador.predict(prob_cruda) if calibrador is not None else prob_cruda

def evaluar_cartera(
    df_original,
    preprocesador,
    modelo,
    umbral_riesgo_alto=0.65,
    umbral_estable=0.30,
    saldo_vip=100000.0,
    costo_vip=250.0,
    efectividad_vip=0.45,
    costo_bono=60.0,
    efectividad_bono=0.30,
    probabilidades=None
):
    """
    Cruza las probabilidades de la red neuronal con la regla financiera
    y clasifica a los clientes en los Segmentos A, B, C y D del informe.
    Permite variar costos y efectividades desde el Simulador de Streamlit.
    Si se pasan 'probabilidades' (calculadas con predecir_probabilidades) no se vuelve a
    ejecutar la red neuronal.
    """
    df = df_original.copy()

    # 1. Probabilidad de fuga con la Red Neuronal (o la recibida ya calculada)
    if probabilidades is None:
        probabilidades = predecir_probabilidades(df, preprocesador, modelo)
    df['Probabilidad_Fuga'] = np.asarray(probabilidades)

    # 2. Calculamos el margen anual de cada cliente
    df['Margen_Cliente'] = calcular_margen_cliente(df)

    # 3. Aplicamos la formula oficial de Ganancia Neta Esperada para ambos incentivos:
    # Ganancia Neta = (Probabilidad de Fuga * Efectividad * Margen) - Costo
    ganancia_vip = (df['Probabilidad_Fuga'] * efectividad_vip * df['Margen_Cliente']) - costo_vip
    ganancia_bono = (df['Probabilidad_Fuga'] * efectividad_bono * df['Margen_Cliente']) - costo_bono

    # 4. Asignamos Segmentos, Decisiones, Incentivos y Costos de forma vectorizada
    prob = df['Probabilidad_Fuga'].to_numpy()
    saldo = df['Balance'].to_numpy()
    g_vip = ganancia_vip.to_numpy()
    g_bono = ganancia_bono.to_numpy()

    # Segmento D: Cliente Estable (< umbral_estable de probabilidad)
    es_d = prob < umbral_estable
    # Segmento A: riesgo alto, saldo superior y plan VIP rentable (ganancia neta > 0)
    es_a_seg = ~es_d & (prob >= umbral_riesgo_alto) & (saldo >= saldo_vip) & (g_vip > 0)
    # Segmento B: riesgo relevante y la bonificacion transaccional da ganancia neta > 0
    es_b_seg = ~es_d & ~es_a_seg & (g_bono > 0)
    # Segmento C: en riesgo pero ningun incentivo da ganancia neta (deficitario / saldo bajo)
    # -> corresponde a todo lo que no es D, A ni B

    condiciones = [es_d, es_a_seg, es_b_seg]
    df['Segmento'] = np.select(
        condiciones, ['D: Estable', 'A: Alto Valor / Alto Riesgo', 'B: Medio / Riesgo Moderado'],
        default='C: No Rentable')
    df['Decision_Sistema'] = np.select(
        condiciones, ['Sin Acción Requerida', 'Retención Prioritaria', 'Retención Automatizada'],
        default='Salida Permitida')
    df['Incentivo_Asignado'] = np.select(
        condiciones, ['Mantenimiento Ordinario', 'Gestor VIP y Tarifa Plana', 'Bonificación Transaccional'],
        default='Sin intervención')
    df['Costo_Incentivo'] = np.select(condiciones, [0.0, costo_vip, costo_bono], default=0.0)
    df['Ganancia_Neta_Esperada'] = np.select(condiciones, [0.0, g_vip, g_bono], default=0.0)

    # 5. Rescate esperado: dinero que se espera retener = Prob. de fuga x Efectividad x Saldo.
    # Solo aplica a los clientes que el sistema decide intervenir (Segmentos A y B).
    es_a = df['Segmento'].str.startswith('A')
    es_b = df['Segmento'].str.startswith('B')
    df['Rescate_Esperado'] = np.select(
        [es_a, es_b],
        [df['Probabilidad_Fuga'] * efectividad_vip * df['Balance'],
         df['Probabilidad_Fuga'] * efectividad_bono * df['Balance']],
        default=0.0
    )

    return df

def obtener_resumen_ejecutivo(df_evaluado, umbral_riesgo_alto=0.65):
    """
    Calcula exactamente los KPIs que piden las secciones 1 y 3 de la plantilla del informe.
    """
    total_clientes = len(df_evaluado)
    
    # Clientes en riesgo relevante (>= umbral_riesgo_alto)
    en_riesgo = df_evaluado[df_evaluado['Probabilidad_Fuga'] >= umbral_riesgo_alto]
    capital_riesgo_critico = en_riesgo['Balance'].sum()

    # Clientes rentables que el sistema decide intervenir (Segmentos A y B)
    intervenidos = df_evaluado[df_evaluado['Segmento'].isin(['A: Alto Valor / Alto Riesgo', 'B: Medio / Riesgo Moderado'])]
    capital_rescatable = intervenidos['Balance'].sum()
    inversion_total = intervenidos['Costo_Incentivo'].sum()
    beneficio_neto_total = intervenidos['Ganancia_Neta_Esperada'].sum()
    rescate_esperado = intervenidos['Rescate_Esperado'].sum()

    # ROI = (Beneficio Neto / Inversion Total) * 100
    roi = (beneficio_neto_total / inversion_total * 100) if inversion_total > 0 else 0.0

    # Tabla resumen por segmento para el Punto 3 del informe
    tabla_segmentos = df_evaluado.groupby('Segmento').agg(
        Clientes=('CustomerId', 'count'),
        Prob_Promedio=('Probabilidad_Fuga', 'mean'),
        Saldo_Promedio=('Balance', 'mean'),
        Balance_Total=('Balance', 'sum'),
        Costo_Unitario=('Costo_Incentivo', 'mean'),
        Inversion_Total=('Costo_Incentivo', 'sum'),
        Rescate_Esperado=('Rescate_Esperado', 'sum'),
        Retorno_Neto_Total=('Ganancia_Neta_Esperada', 'sum')
    ).reset_index()

    return {
        'total_clientes': total_clientes,
        'capital_riesgo_critico': capital_riesgo_critico,
        'capital_rescatable': capital_rescatable,
        'rescate_esperado': rescate_esperado,
        'inversion_total': inversion_total,
        'beneficio_neto_total': beneficio_neto_total,
        'roi_porcentaje': roi,
        'tabla_segmentos': tabla_segmentos
    }

if __name__ == '__main__':
    print("Probando el Motor Financiero con toda la cartera del banco...")
    df_raw = pd.read_csv('Customer-Churn-Records.csv')
    prep, mod = cargar_artefactos()
    
    df_res = evaluar_cartera(df_raw, prep, mod)
    resumen = obtener_resumen_ejecutivo(df_res)
    
    print("\n--- RESUMEN DE IMPACTO FINANCIERO ---")
    print(f"Volumen Total de Clientes Analizados: {resumen['total_clientes']:,}")
    print(f"Monto Total de Capital en Riesgo Critico: ${resumen['capital_riesgo_critico']:,.2f}")
    print(f"Saldo de Clientes a Intervenir (Segmentos A+B): ${resumen['capital_rescatable']:,.2f}")
    print(f"Rescate Esperado (Prob x Efectividad x Saldo): ${resumen['rescate_esperado']:,.2f}")
    print(f"Inversion Total en Campanas de Fidelizacion: ${resumen['inversion_total']:,.2f}")
    print(f"Ganancia Neta Esperada (Despues de Costos): ${resumen['beneficio_neto_total']:,.2f}")
    print(f"Retorno Sobre la Inversion (ROI): {resumen['roi_porcentaje']:.2f}%")
    
    print("\n--- MATRIZ DE DECISION COMERCIAL ---")
    print(resumen['tabla_segmentos'].to_string(index=False))
