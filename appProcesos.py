import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import datetime
import io

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
    """Carga el archivo de Excel o CSV."""
    
    if file.name.endswith(('.xlsx', '.xlsm')):
        try:
            df = pd.read_excel(io.BytesIO(file.getvalue()), engine='openpyxl')
        except Exception as e:
            st.error(f"Error al leer el archivo de Excel. Asegúrate de que el formato sea correcto. Error: {e}")
            return None
    elif file.name.endswith('.csv'):
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Error al leer el archivo CSV. Error: {e}")
            return None
    else:
        st.error("Formato de archivo no soportado. Por favor sube un archivo CSV, XLSX o XLSM.")
        return None
        
    return df.copy() # Devolvemos una copia para evitar problemas de SettingWithCopyWarning

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
    
    # 2. Agregar Regresor Adicional
    if regressor_col:
        m.add_regressor(regressor_col)
        
    m.fit(df)
    
    # 3. Crear el DataFrame de futuro
    future = m.make_future_dataframe(periods=n_days)
    
    # 4. Asegurarse de tener los valores del regresor en el futuro
    if regressor_col:
        # Usamos el valor promedio histórico del precio para la predicción futura.
        future[regressor_col] = df[regressor_col].mean() 
    
    # 5. Realizar la predicción
    forecast = m.predict(future)
    
    return m, forecast

# --- Sidebar para Carga y Configuración de Datos ---
with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo CSV, XLSX o XLSM", type=["csv", "xlsx", "xlsm"])

    df = None
    if uploaded_file is not None:
        df_original = load_data(uploaded_file)
        
        if df_original is not None:
            st.success("Archivo cargado correctamente. Ahora asigna las columnas.")
            
            # --- Selectores de Columna ---
            st.header("2. Asignación de Columnas")
            column_names = df_original.columns.tolist()
            
            # Selector de Columna de Fecha (ds)
            ds_col = st.selectbox("Columna de Fecha (ds):", 
                                  options=column_names, 
                                  index=0) # Asume la primera columna como la fecha
            
            # Selector de Columna de Cantidad (y)
            y_col = st.selectbox("Columna de Cantidad (y - Demanda):", 
                                  options=column_names, 
                                  index=min(1, len(column_names) - 1)) # Asume la segunda columna
            
            # Selector de Columna de Precio (Regresor)
            st.subheader("Regresor Adicional (Precio)")
            use_regressor = st.checkbox("Usar una columna de Precio como regresor", value=True)
            
            regressor_col = None
            if use_regressor:
                price_col = st.selectbox("Columna de Precio:", 
                                         options=['(Ninguno)'] + column_names, 
                                         index=min(2, len(column_names))) # Asume la tercera columna
                if price_col != '(Ninguno)':
                    regressor_col = 'precio_regressor' # Nombre temporal para el modelo
            
            # --- Preprocesamiento y Renombre ---
            try:
                # Renombrar las columnas al formato Prophet
                df = df_original.rename(columns={ds_col: 'ds', y_col: 'y'})
                
                # Conversión de tipos de datos
                df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                
                # Manejar el Regresor
                if regressor_col:
                    df = df.rename(columns={price_col: regressor_col})
                    df[regressor_col] = pd.to_numeric(df[regressor_col], errors='coerce')
                    df = df.dropna(subset=['ds', 'y', regressor_col])
                else:
                    df = df.dropna(subset=['ds', 'y'])
                
                # Mostrar los datos preprocesados
                st.subheader("Datos Preprocesados (Prophet Ready)")
                st.dataframe(df[['ds', 'y'] + ([regressor_col] if regressor_col else [])].head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al asignar o convertir columnas. Revisa tus selecciones. Detalle: {e}")
                df = None
            
            
            # --- Configuración del Pronóstico ---
            if df is not None and not df.empty:
                st.header("3. Configuración del Pronóstico")
                st.subheader("Horizonte de Pronóstico")
                n_days = st.slider("Días a Pronosticar en el Futuro:", 
                                   min_value=7, max_value=365, value=90, step=7)


# --- Contenido Principal ---
if df is not None and not df.empty:
    
    st.header("Visualización de la Demanda Histórica")
    fig_hist = px.line(df, x='ds', y='y', title='Demanda Histórica (Cantidad de Servicios)')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.divider()

    st.header("4. Ejecución del Pronóstico")
    if st.button("Generar Pronóstico"):
        
        # Validar si el regresor se usó, se entrena con su nombre temporal
        regressor_name_in_model = regressor_col if regressor_col else None
        
        with st.spinner(f"Entrenando modelo Prophet y prediciendo {n_days} días..."):
            m, forecast = train_prophet_model(df, n_days, regressor_name_in_model)
        
        st.success("¡Pronóstico Generado!")
        
        # --- Resultados del Pronóstico ---
        
        st.subheader(f"Predicción para los Próximos {n_days} Días")
        
        # 1. Gráfico del Pronóstico (Prophet Plotly)
        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(title='Pronóstico de Demanda vs. Real', 
                           xaxis_title='Fecha', 
                           yaxis_title=y_col) # Usamos el nombre original de la columna
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Componentes del Pronóstico")
        
        # 2. Gráfico de Componentes (Prophet Plotly)
        fig2 = plot_components_plotly(m, forecast)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Datos Detallados del Pronóstico")
        
        # 3. Datos del Pronóstico
        future_forecast = forecast[forecast['ds'] > df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_forecast.columns = ['Fecha', f'Demanda_Estimada ({y_col})', 'Límite_Inferior', 'Límite_Superior']
        
        st.dataframe(future_forecast.head(10), use_container_width=True)

        # Botón de Descarga
        csv = future_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Pronóstico como CSV",
            data=csv,
            file_name=f'pronostico_demanda_{datetime.date.today()}.csv',
            mime='text/csv',
        )

else:
    st.info("Por favor, sube un archivo (CSV, XLSX o XLSM) en la barra lateral y asigna las columnas para comenzar.")
