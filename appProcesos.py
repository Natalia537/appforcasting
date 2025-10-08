import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import datetime
import io  # Importar la biblioteca 'io'

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Pronóstico de Demanda de Servicios con Prophet y Streamlit",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Demand Forecasting App 📈")
st.markdown("Usa Prophet para predecir la demanda (cantidad de servicios) basándose en la fecha y el precio.")

# --- Funciones de Cache ---
@st.cache_data
def load_data(file):
    """Carga y preprocesa el archivo de Excel o CSV."""
    
    # Manejar archivos XLSM/XLSX
    if file.name.endswith(('.xlsx', '.xlsm')):
        try:
            # Lee el archivo de bytes y lo pasa a pandas
            df = pd.read_excel(io.BytesIO(file.getvalue()), engine='openpyxl')
        except Exception as e:
            st.error(f"Error al leer el archivo de Excel. Asegúrate de que el formato sea correcto. Error: {e}")
            st.stop()
    # Manejar archivos CSV
    elif file.name.endswith('.csv'):
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Error al leer el archivo CSV. Error: {e}")
            st.stop()
    else:
        st.error("Formato de archivo no soportado. Por favor sube un archivo CSV, XLSX o XLSM.")
        st.stop()

    required_cols = ['ds', 'y']
    
    # Validación de columnas
    if not all(col in df.columns for col in required_cols):
        st.error(f"El archivo debe contener las columnas 'ds' (fecha) y 'y' (cantidad).")
        st.stop()
        
    # Conversión de tipos
    try:
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
    except Exception as e:
        st.error(f"Error al convertir tipos de datos. Asegúrate de que las columnas 'ds' y 'y' contengan datos válidos. Error: {e}")
        st.stop()
    
    # Verificar si la columna 'precio' existe para usarla como regresor
    if 'precio' in df.columns:
        df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
        df = df.dropna(subset=['precio']).copy()
    
    return df

@st.cache_resource
def train_prophet_model(df, n_days, regressor_col=None):
    """Entrena el modelo Prophet y realiza la predicción."""
    
    # 1. Crear y entrenar el modelo
    m = Prophet(
        seasonality_mode='multiplicative',
        daily_seasonality=False,
        weekly_seasonality='auto',
        yearly_seasonality='auto'
    )
    
    # 2. Agregar Regresor Adicional (Precio)
    if regressor_col:
        m.add_regressor(regressor_col)
        
    m.fit(df)
    
    # 3. Crear el DataFrame de futuro
    future = m.make_future_dataframe(periods=n_days)
    
    # 4. Asegurarse de tener los valores del regresor en el futuro
    if regressor_col:
        future[regressor_col] = df[regressor_col].mean() 
    
    # 5. Realizar la predicción
    forecast = m.predict(future)
    
    return m, forecast

# --- Sidebar para Carga y Configuración de Datos ---
with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo CSV, XLSX o XLSM", type=["csv", "xlsx", "xlsm"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("Archivo cargado y preprocesado correctamente!")
        st.dataframe(df.head(), use_container_width=True)
        
        st.header("2. Configuración del Modelo")
        
        st.subheader("Horizonte de Pronóstico")
        n_days = st.slider("Días a Pronosticar en el Futuro:", 
                           min_value=7, max_value=365, value=90, step=7)

        st.subheader("Regresores Adicionales")
        use_price_regressor = st.checkbox("Usar la columna 'precio' como regresor", value=('precio' in df.columns))
        
        regressor_col = 'precio' if use_price_regressor and 'precio' in df.columns else None

# --- Contenido Principal ---
if 'df' in locals():
    
    st.header("Visualización de la Demanda Histórica")
    fig_hist = px.line(df, x='ds', y='y', title='Demanda Histórica (Cantidad de Servicios)')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.divider()

    st.header("3. Ejecución del Pronóstico")
    if st.button("Generar Pronóstico"):
        with st.spinner(f"Entrenando modelo Prophet y prediciendo {n_days} días..."):
            m, forecast = train_prophet_model(df, n_days, regressor_col)
        
        st.success("¡Pronóstico Generado!")
        
        st.subheader(f"Predicción para los Próximos {n_days} Días")
        
        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(title='Pronóstico de Demanda vs. Real', 
                           xaxis_title='Fecha (ds)', 
                           yaxis_title='Demanda Estimada (yhat)')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Componentes del Pronóstico")
        
        fig2 = plot_components_plotly(m, forecast)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Datos Detallados del Pronóstico")
        
        future_forecast = forecast[forecast['ds'] > df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_forecast.columns = ['Fecha', 'Demanda_Estimada', 'Límite_Inferior', 'Límite_Superior']
        
        st.dataframe(future_forecast.head(10), use_container_width=True)

        csv = future_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Pronóstico como CSV",
            data=csv,
            file_name=f'pronostico_demanda_{datetime.date.today()}.csv',
            mime='text/csv',
        )

else:
    st.info("Por favor, sube un archivo CSV, XLSX o XLSM en la barra lateral para comenzar.")
