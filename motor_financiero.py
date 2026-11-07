import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Cargamos los artefactos entrenados una sola vez (Canal de Inferencia)
def cargar_artefactos():
    preprocesador = joblib.load('preprocesador.joblib')
    modelo = tf.keras.models.load_model('modelo_churn.keras')
    return preprocesador, modelo

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
    efectividad_bono=0.30
):
    """
    Cruza las probabilidades de la red neuronal con la regla financiera
    y clasifica a los clientes en los Segmentos A, B, C y D del informe.
    Permite variar costos y efectividades desde el Simulador de Streamlit.
    """
    df = df_original.copy()

    # 1. Preprocesamos los datos y predecimos la probabilidad con la Red Neuronal
    cols_entrada = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Satisfaction Score', 'Card Type', 'Point Earned'
    ]
    X_proc = preprocesador.transform(df[cols_entrada])
    df['Probabilidad_Fuga'] = modelo.predict(X_proc, verbose=0).flatten()

    # 2. Calculamos el margen anual de cada cliente
    df['Margen_Cliente'] = calcular_margen_cliente(df)

    # 3. Aplicamos la formula oficial de Ganancia Neta Esperada para ambos incentivos:
    # Ganancia Neta = (Probabilidad de Fuga * Efectividad * Margen) - Costo
    ganancia_vip = (df['Probabilidad_Fuga'] * efectividad_vip * df['Margen_Cliente']) - costo_vip
    ganancia_bono = (df['Probabilidad_Fuga'] * efectividad_bono * df['Margen_Cliente']) - costo_bono

    # 4. Asignamos Segmentos, Decisiones, Incentivos y Costos de forma vectorizada
    segmentos = []
    decisiones = []
    incentivos = []
    costos = []
    ganancias_netas = []

    for i in range(len(df)):
        prob = df['Probabilidad_Fuga'].iloc[i]
        saldo = df['Balance'].iloc[i]
        g_vip = ganancia_vip.iloc[i]
        g_bono = ganancia_bono.iloc[i]

        # Segmento D: Cliente Estable (< 0.30 de probabilidad)
        if prob < umbral_estable:
            segmentos.append('D: Estable')
            decisiones.append('Sin Acción Requerida')
            incentivos.append('Mantenimiento Ordinario')
            costos.append(0.0)
            ganancias_netas.append(0.0)

        # Clientes con riesgo relevante (>= 0.30)
        else:
            # Si tiene alto riesgo (>= 0.65), saldo superior y el plan VIP es rentable (> 0)
            if prob >= umbral_riesgo_alto and saldo >= saldo_vip and g_vip > 0:
                segmentos.append('A: Alto Valor / Alto Riesgo')
                decisiones.append('Retención Prioritaria')
                incentivos.append('Gestor VIP y Tarifa Plana')
                costos.append(costo_vip)
                ganancias_netas.append(g_vip)

            # Si tiene riesgo moderado/alto y la bonificacion transaccional da ganancia neta > 0
            elif g_bono > 0:
                segmentos.append('B: Medio / Riesgo Moderado')
                decisiones.append('Retención Automatizada')
                incentivos.append('Bonificación Transaccional')
                costos.append(costo_bono)
                ganancias_netas.append(g_bono)

            # Si está en riesgo pero da perdida neta tras deducir el incentivo (Deficitario / Saldo Bajo)
            else:
                segmentos.append('C: No Rentable')
                decisiones.append('Salida Permitida')
                incentivos.append('Sin intervención')
                costos.append(0.0)
                ganancias_netas.append(0.0)

    df['Segmento'] = segmentos
    df['Decision_Sistema'] = decisiones
    df['Incentivo_Asignado'] = incentivos
    df['Costo_Incentivo'] = costos
    df['Ganancia_Neta_Esperada'] = ganancias_netas

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
        Retorno_Neto_Total=('Ganancia_Neta_Esperada', 'sum')
    ).reset_index()

    return {
        'total_clientes': total_clientes,
        'capital_riesgo_critico': capital_riesgo_critico,
        'capital_rescatable': capital_rescatable,
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
    print(f"Capital Rescatable Estimado (Clientes Rentables): ${resumen['capital_rescatable']:,.2f}")
    print(f"Inversion Total en Campanas de Fidelizacion: ${resumen['inversion_total']:,.2f}")
    print(f"Ganancia Neta Esperada (Despues de Costos): ${resumen['beneficio_neto_total']:,.2f}")
    print(f"Retorno Sobre la Inversion (ROI): {resumen['roi_porcentaje']:.2f}%")
    
    print("\n--- MATRIZ DE DECISION COMERCIAL ---")
    print(resumen['tabla_segmentos'].to_string(index=False))
