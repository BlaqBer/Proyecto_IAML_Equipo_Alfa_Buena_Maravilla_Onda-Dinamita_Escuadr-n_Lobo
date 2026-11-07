import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from motor_financiero import cargar_artefactos, evaluar_cartera, obtener_resumen_ejecutivo

# Configuracion general de la pagina web
st.set_page_config(
    page_title="Deep Retention | Sistema Bancario",
    page_icon="🏦",
    layout="wide"
)

# Guardamos en cache el modelo y el dataset para que la app vuele al mover los sliders
@st.cache_resource
def iniciar_sistema():
    return cargar_artefactos()

@st.cache_data
def cargar_datos_base():
    return pd.read_csv('Customer-Churn-Records.csv')

@st.cache_data
def cargar_metricas_modelo():
    if os.path.exists('metricas_tecnicas.joblib'):
        return joblib.load('metricas_tecnicas.joblib')
    return None

preprocesador, modelo = iniciar_sistema()
df_original = cargar_datos_base()
metricas_tec = cargar_metricas_modelo()

# --- BARRA LATERAL: SIMULADOR DE PARAMETROS EN VIVO ---
st.sidebar.header("Simulador de Parámetros")
st.sidebar.markdown("Hola internauta, solo toca aqui si quieres ajustar los costos y umbrales para simular el impacto en la rentabilidad neta del banco, de no saber lo que haces no toques (es consejo, no advertencia):")

umbral_critico = st.sidebar.slider("Umbral Riesgo Alto (Probabilidad)", 0.50, 0.90, 0.65, 0.05)
umbral_estable = st.sidebar.slider("Umbral Cliente Estable (< Prob)", 0.10, 0.45, 0.30, 0.05)
saldo_min_vip = st.sidebar.number_input("Saldo Mínimo Cuenta VIP ($)", value=100000.0, step=10000.0)

st.sidebar.subheader("Plan A: Gestor VIP y Tarifa Plana")
costo_vip = st.sidebar.slider("Costo Unitario Plan VIP ($)", 50.0, 600.0, 250.0, 10.0)
efectividad_vip = st.sidebar.slider("Efectividad Estimada VIP (%)", 10, 80, 45, 5) / 100.0

st.sidebar.subheader("Plan B: Bonificación Transaccional")
costo_bono = st.sidebar.slider("Costo Unitario Bono ($)", 10.0, 200.0, 60.0, 5.0)
efectividad_bono = st.sidebar.slider("Efectividad Estimada Bono (%)", 5, 60, 30, 5) / 100.0

# Evaluamos toda la cartera en tiempo real con los parametros de los sliders
df_evaluado = evaluar_cartera(
    df_original, preprocesador, modelo,
    umbral_riesgo_alto=umbral_critico,
    umbral_estable=umbral_estable,
    saldo_vip=saldo_min_vip,
    costo_vip=costo_vip,
    efectividad_vip=efectividad_vip,
    costo_bono=costo_bono,
    efectividad_bono=efectividad_bono
)
resumen = obtener_resumen_ejecutivo(df_evaluado, umbral_riesgo_alto=umbral_critico)

# Calculamos tambien el capital rescatable exclusivo de riesgo critico (Segmento A)
cap_rescatable_critico = df_evaluado[df_evaluado['Segmento'] == 'A: Alto Valor / Alto Riesgo']['Balance'].sum()

# --- ENCABEZADO PRINCIPAL ---
st.title("Sistema Deep Retention: Los clientes que se van, cuestan dinero al banco")
st.markdown(
    "**Directriz Operativa:** Únicamente se autoriza activar un plan de retención comercial si el cliente "
    "representa una **ganancia potencial neta positiva** para el banco tras deducir el costo del incentivo. Vamos, en pocas palabras, si el cliente es rentable y se va, hay que retenerlo, si no es rentable, en su casa esta mejor ;) "
)

# Creamos las pestañas de navegacion
tab1, tab2, tab3, tab4 = st.tabs([
    "Panel de Cartera (Dashboard)",
    "Buscador y Evaluador de Clientes",
    "Análisis del Simulador Financiero",
    "Datos para el Informe Ejecutivo"
])

# ==============================================================================
# PESTAÑA 1: PANEL DE CARTERA (DASHBOARD)
# ==============================================================================
with tab1:
    st.subheader("Monitoreo Consolidado de Cartera y Rentabilidad")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Clientes Analizados", f"{resumen['total_clientes']:,}")
    col2.metric("Capital en Riesgo Crítico", f"${resumen['capital_riesgo_critico']:,.0f}")
    col3.metric("Capital Total a Rescatar (A+B)", f"${resumen['capital_rescatable']:,.0f}")
    col4.metric("Inversión en Retención", f"${resumen['inversion_total']:,.0f}")
    col5.metric("Ganancia Neta / ROI", f"${resumen['beneficio_neto_total']:,.0f}", f"ROI: {resumen['roi_porcentaje']:.1f}%")

    st.markdown("---")
    
    c_graf1, c_graf2 = st.columns(2)
    with c_graf1:
        fig_seg = px.bar(
            resumen['tabla_segmentos'],
            x='Segmento', y='Clientes', color='Segmento',
            text='Clientes',
            title="Distribución de Clientes por Segmento de Decisión",
            color_discrete_sequence=['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c']
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with c_graf2:
        fig_bal = px.pie(
            resumen['tabla_segmentos'],
            names='Segmento', values='Balance_Total',
            title="Proporción de Balance Bancario ($) por Segmento",
            hole=0.4
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    st.subheader("Matriz Consolidada de Decisión Comercial")
    df_mostrar_seg = resumen['tabla_segmentos'].copy()
    df_mostrar_seg['Prob_Promedio'] = (df_mostrar_seg['Prob_Promedio'] * 100).round(2).astype(str) + '%'
    df_mostrar_seg['Saldo_Promedio'] = df_mostrar_seg['Saldo_Promedio'].apply(lambda x: f"${x:,.2f}")
    df_mostrar_seg['Balance_Total'] = df_mostrar_seg['Balance_Total'].apply(lambda x: f"${x:,.2f}")
    df_mostrar_seg['Costo_Unitario'] = df_mostrar_seg['Costo_Unitario'].apply(lambda x: f"${x:,.2f}")
    df_mostrar_seg['Inversion_Total'] = df_mostrar_seg['Inversion_Total'].apply(lambda x: f"${x:,.2f}")
    df_mostrar_seg['Retorno_Neto_Total'] = df_mostrar_seg['Retorno_Neto_Total'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(df_mostrar_seg, use_container_width=True, hide_index=True)

# ==============================================================================
# PESTAÑA 2: BUSCADOR POR CLIENTE Y SIMULACION DE CLIENTE NUEVO
# ==============================================================================
with tab2:
    modo_consulta = st.radio(
        "Seleccione el modo de consulta para el ejecutivo:",
        ["Buscar Cliente Existente en Base de Datos", "Ingresar Datos de un Cliente Nuevo (Simulación Individual)"],
        horizontal=True
    )

    if modo_consulta == "Buscar Cliente Existente en Base de Datos":
        st.subheader("Consulta Individual de Cartera")
        # Creamos una etiqueta facil de buscar con ID y Apellido
        opciones = df_evaluado['CustomerId'].astype(str) + " - " + df_evaluado['Surname'] + " (" + df_evaluado['Segmento'] + ")"
        seleccion = st.selectbox("Escriba el ID o Apellido del cliente:", opciones)
        id_elegido = int(seleccion.split(" - ")[0])
        
        cliente = df_evaluado[df_evaluado['CustomerId'] == id_elegido].iloc[0]

    else:
        st.subheader("Evaluación en Tiempo Real de Cliente Nuevo")
        f1, f2, f3, f4 = st.columns(4)
        score_n = f1.number_input("Credit Score", 300, 850, 650)
        pais_n = f2.selectbox("País (Geography)", ["France", "Germany", "Spain"])
        genero_n = f3.selectbox("Género", ["Female", "Male"])
        edad_n = f4.number_input("Edad", 18, 95, 42)

        f5, f6, f7, f8 = st.columns(4)
        tenure_n = f5.number_input("Años en el banco (Tenure)", 0, 10, 3)
        saldo_n = f6.number_input("Saldo en Cuenta ($)", 0.0, 300000.0, 125000.0, step=5000.0)
        prod_n = f7.selectbox("Número de Productos", [1, 2, 3, 4], index=0)
        activo_n = f8.selectbox("¿Es Miembro Activo?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")

        f9, f10, f11, f12 = st.columns(4)
        tc_n = f9.selectbox("¿Tiene Tarjeta de Crédito?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
        salario_n = f10.number_input("Salario Estimado ($)", 0.0, 250000.0, 95000.0, step=5000.0)
        satisf_n = f11.slider("Puntaje de Satisfacción", 1, 5, 3)
        card_n = f12.selectbox("Tipo de Tarjeta", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])

        nuevo_df = pd.DataFrame([{
            'CustomerId': 99999999, 'Surname': 'Cliente Evaluado',
            'CreditScore': score_n, 'Geography': pais_n, 'Gender': genero_n,
            'Age': edad_n, 'Tenure': tenure_n, 'Balance': saldo_n,
            'NumOfProducts': prod_n, 'HasCrCard': tc_n, 'IsActiveMember': activo_n,
            'EstimatedSalary': salario_n, 'Satisfaction Score': satisf_n,
            'Card Type': card_n, 'Point Earned': 450
        }])
        cliente = evaluar_cartera(
            nuevo_df, preprocesador, modelo,
            umbral_riesgo_alto=umbral_critico, umbral_estable=umbral_estable,
            saldo_vip=saldo_min_vip, costo_vip=costo_vip, efectividad_vip=efectividad_vip,
            costo_bono=costo_bono, efectividad_bono=efectividad_bono
        ).iloc[0]

    # Mostramos la tarjeta de resultados del cliente seleccionado
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    prob_c = cliente['Probabilidad_Fuga']
    k1.metric("Probabilidad de Abandono (Red)", f"{prob_c*100:.1f}%")
    k2.metric("Balance en Cuenta", f"${cliente['Balance']:,.2f}")
    k3.metric("Margen Anual Estimado", f"${cliente['Margen_Cliente']:,.2f}")
    k4.metric("Ganancia Neta de Retención", f"${cliente['Ganancia_Neta_Esperada']:,.2f}", f"Costo: -${cliente['Costo_Incentivo']:,.0f}")

    # Alerta visual segun la decision comercial
    if cliente['Decision_Sistema'] == 'Retención Prioritaria':
        st.error(f"🚨 **ACCIÓN SUGERIDA: {cliente['Decision_Sistema']} ({cliente['Segmento']})**  \n"
                 f"**Incentivo Autorizado:** {cliente['Incentivo_Asignado']} (Costo: ${cliente['Costo_Incentivo']:,.2f}) — "
                 f"El cliente presenta alto riesgo de fuga y alta rentabilidad patrimonial.")
    elif cliente['Decision_Sistema'] == 'Retención Automatizada':
        st.warning(f"⚠️ **ACCIÓN SUGERIDA: {cliente['Decision_Sistema']} ({cliente['Segmento']})**  \n"
                   f"**Incentivo Autorizado:** {cliente['Incentivo_Asignado']} (Costo: ${cliente['Costo_Incentivo']:,.2f}) — "
                   f"Retención rentable mediante campaña digital automatizada.")
    elif cliente['Decision_Sistema'] == 'Salida Permitida':
        st.info(f"🛑 **ACCIÓN SUGERIDA: {cliente['Decision_Sistema']} ({cliente['Segmento']})**  \n"
                f"**Incentivo Bloqueado ($0.00):** El cliente tiene alto riesgo de abandono ({prob_c*100:.1f}%), "
                f"pero su saldo (${cliente['Balance']:,.2f}) no compensa el costo del beneficio comercial.")
    else:
        st.success(f"✅ **ACCIÓN SUGERIDA: {cliente['Decision_Sistema']} ({cliente['Segmento']})**  \n"
                   f"Cliente estable con baja probabilidad de fuga. Mantener gestión ordinaria sin gasto adicional.")

# ==============================================================================
# PESTAÑA 3: ANALISIS DEL SIMULADOR FINANCIERO
# ==============================================================================
with tab3:
    st.subheader("Simulación de Rentabilidad vs. Probabilidad de Fuga")
    st.markdown("Mueve los controles del menú izquierdo para observar cómo cambian los umbrales de rentabilidad en toda la base:")
    
    # Muestra de 1,500 clientes para que el grafico de dispersion sea fluido
    muestra_graf = df_evaluado.sample(n=min(1500, len(df_evaluado)), random_state=42).copy()
    
    # Plotly no puede dibujar circulos con radio negativo (clientes deficitarios de -$35),
    # asi que fijamos un tamaño visual minimo de 15 para el dibujo sin alterar el dato real
    muestra_graf['Tamano_Burbuja'] = muestra_graf['Margen_Cliente'].clip(lower=15.0)

    fig_scatter = px.scatter(
        muestra_graf,
        x="Probabilidad_Fuga", y="Balance",
        color="Segmento", size="Tamano_Burbuja",
        hover_data=['CustomerId', 'Surname', 'Decision_Sistema', 'Margen_Cliente', 'Ganancia_Neta_Esperada'],
        title="Mapa de Decisión: Probabilidad de la Red Neuronal vs. Valor Patrimonial (Balance)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# PESTAÑA 4: DATOS LISTOS PARA TUS COMPAÑEROS (INFORME EJECUTIVO)
# ==============================================================================
with tab4:
    st.subheader("📄 Plantilla Autocompletada para el Informe Ejecutivo")
    st.markdown("Estos datos se calculan directamente del modelo y la cartera para completar las secciones **1, 2 y 3** del PDF:")

    col_inf1, col_inf2 = st.columns(2)
    with col_inf1:
        st.markdown("#### 1. Resumen de Impacto Financiero y Retorno")
        st.write(f"- **Volumen Total de Clientes Analizados:** `{resumen['total_clientes']:,}`")
        st.write(f"- **Monto Total de Capital en Riesgo Crítico (Prob ≥ {umbral_critico}):** `${resumen['capital_riesgo_critico']:,.2f}`")
        st.write(f"- **Capital Rescatable Estimado (Crítico Segmento A):** `${cap_rescatable_critico:,.2f}` *(Total A+B: `${resumen['capital_rescatable']:,.2f}`)*")
        st.write(f"- **Inversión Total en Campañas de Fidelización:** `${resumen['inversion_total']:,.2f}`")
        st.write(f"- **Beneficio Neto Esperado (tras deducir incentivos):** `${resumen['beneficio_neto_total']:,.2f}`")
        st.write(f"- **Retorno Sobre la Inversión (ROI):** `{resumen['roi_porcentaje']:.2f}%`")

        if metricas_tec:
            st.markdown("#### 2. Diagnóstico Técnico de la Red Neuronal")
            st.write(f"- **Arquitectura:** Perceptrón Multicapa (MLP) con 2 capas ocultas (64 y 32 neuronas, ReLU, BatchNorm, Dropout 0.3/0.2) y salida Sigmoide, entrenado con optimizador Adam y pesos balanceados.")
            st.write(f"- **ROC-AUC:** `{metricas_tec['roc_auc']:.4f}` | **Precision-Recall AUC:** `{metricas_tec['pr_auc']:.4f}` | **Precisión de Calibración:** `{metricas_tec['calibracion']:.4f}`")
            st.write("**Top 3 Factores Determinantes de Abandono:**")
            for idx, fac in enumerate(metricas_tec['top_factores'][:3], 1):
                st.write(f"  {idx}. `{fac['Variable']}` (Impacto medio en probabilidad: `{fac['Impacto_SHAP']:.4f}`)")

    with col_inf2:
        st.markdown("#### Gráfica de Explicabilidad de la Red (Importancia de Variables)")
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption="Factores determinantes de abandono identificados en el modelo")
        else:
            st.info("Ejecuta 'python generar_explicabilidad.py' para visualizar la gráfica aquí.")