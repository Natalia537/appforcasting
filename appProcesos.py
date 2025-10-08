import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go
import datetime
import io
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Comparador de Pronóstico de Demanda",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Demand Forecasting Dashboard 📈📊")
st.markdown("Compara 6 métodos de pronóstico (Prophet y 5 modelos de Suavizado Exponencial) para encontrar el mejor modelo para tu demanda de servicios.")

# --- Funciones de Carga y Métrica ---

@st.cache_data
def load_data(file):
    """Carga el archivo de Excel o CSV."""
    if file.name.endswith(('.xlsx', '.xlsm')):
        try:
            df = pd.read_excel(io.BytesIO(file.getvalue()), engine='openpyxl')
        except Exception as e:
            st.error(f"Error al leer el archivo de Excel. Error: {e}")
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
    return df.copy()

def calculate_mad_percentage(actual, predicted):
    """Calcula el Porcentaje de Error Absoluto Medio (MAD%)."""
    # Usamos solo los valores no nulos
    actual = np.array(actual[~np.isnan(actual)])
    predicted = np.array(predicted[~np.isnan(predicted)])
    
    # Aseguramos que la división por cero no cause error, usamos 1e-8
    mad = np.mean(np.abs(actual - predicted))
    mean_actual = np.mean(actual)
    
    if mean_actual == 0:
        return 0.0
    
    # Porcentaje de Error Absoluto Medio (Mean Absolute Percentage Error - MAPE)
    # A veces es llamado MAD% o APE, pero MAPE es más estándar.
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return mape

# --- Funciones de Modelado (Prophet y Statsmodels) ---

def run_prophet(df_train, df_full, n_days, regressor_name):
    """Ejecuta el modelo Prophet."""
    try:
        m = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
        if regressor_name:
            m.add_regressor(regressor_name)
        
        m.fit(df_train)
        
        future = m.make_future_dataframe(periods=n_days)
        
        # Asignar el valor promedio del regresor a las fechas futuras
        if regressor_name:
            # Usamos el promedio de la columna en el DataFrame completo, no solo en el de entrenamiento
            future[regressor_name] = df_full[regressor_name].mean()
            
        forecast = m.predict(future)
        
        # Unir la predicción con los datos históricos para calcular el error
        df_join = df_train.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        mape = calculate_mad_percentage(df_join['y'], df_join['yhat'])
        
        return forecast, mape
    except Exception as e:
        return None, f"Error Prophet: {e}"

def run_statsmodels(df_train, n_days, model_type, **kwargs):
    """Ejecuta modelos de suavizado exponencial y promedio móvil."""
    try:
        y_train = df_train.set_index('ds')['y']
        
        if model_type == 'Moving Average':
            window = kwargs.get('window', 7)
            fit = y_train.rolling(window=window).mean()
            # Predicción: el último valor del MA se extiende
            last_ma_value = fit.iloc[-1] if not fit.empty else y_train.mean()
            
            forecast_index = pd.date_range(start=df_train['ds'].max() + pd.Timedelta(days=1), periods=n_days, freq='D')
            forecast_values = np.full(n_days, last_ma_value)
            
            # Crear DataFrame de pronóstico
            forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast_values})
            
            # Calcular MAPE: Comparamos el valor de 'y' con el valor MA calculado
            mape = calculate_mad_percentage(y_train[window:], fit[window:])
            
        elif model_type == 'SES': # Simple Exponential Smoothing
            fit = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(
                smoothing_level=kwargs.get('alpha'), optimized=True
            )
            forecast = fit.predict(start=len(y_train), end=len(y_train) + n_days - 1)
            mape = calculate_mad_percentage(y_train.values, fit.fittedvalues.values)
            forecast_df = pd.DataFrame({'ds': forecast.index, 'yhat': forecast.values})

        elif model_type == 'Holt':
            fit = Holt(y_train, initialization_method="estimated").fit(
                smoothing_level=kwargs.get('alpha'), smoothing_trend=kwargs.get('beta'), optimized=True
            )
            forecast = fit.predict(start=len(y_train), end=len(y_train) + n_days - 1)
            mape = calculate_mad_percentage(y_train.values, fit.fittedvalues.values)
            forecast_df = pd.DataFrame({'ds': forecast.index, 'yhat': forecast.values})

        elif model_type == 'Winter_Add' or model_type == 'Winter_Mult':
            seasonal_periods = kwargs.get('seasonal_periods', 7)
            seasonal_type = 'add' if model_type == 'Winter_Add' else 'mul'
            
            fit = ExponentialSmoothing(
                y_train, 
                seasonal_periods=seasonal_periods, 
                trend=kwargs.get('trend_type'), 
                seasonal=seasonal_type, 
                initialization_method="estimated"
            ).fit(optimized=True)
            
            forecast = fit.predict(start=len(y_train), end=len(y_train) + n_days - 1)
            mape = calculate_mad_percentage(y_train.values, fit.fittedvalues.values)
            forecast_df = pd.DataFrame({'ds': forecast.index, 'yhat': forecast.values})
        
        else:
            return None, "Método no reconocido"
            
        # Aseguramos que 'ds' esté en formato datetime para el merge
        if 'ds' not in forecast_df.columns:
            forecast_df['ds'] = forecast_df.index
        
        return forecast_df, mape
        
    except Exception as e:
        return None, f"Error Statsmodels ({model_type}): {e}"

# --- Sidebar para Carga y Configuración de Datos ---

with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo CSV, XLSX o XLSM", type=["csv", "xlsx", "xlsm"])

    df = None
    if uploaded_file is not None:
        df_original = load_data(uploaded_file)
        
        if df_original is not None:
            st.success("Archivo cargado. Asigna las columnas y configura los modelos.")
            
            # --- Selectores de Columna ---
            st.header("2. Asignación de Columnas")
            column_names = df_original.columns.tolist()
            
            ds_col = st.selectbox("Columna de Fecha (ds):", options=column_names, index=0)
            y_col = st.selectbox("Columna de Cantidad (y - Demanda):", options=column_names, index=min(1, len(column_names) - 1))
            
            # Selector de Columna de Precio (Regresor)
            st.subheader("Regresor Adicional (Precio para Prophet)")
            use_regressor = st.checkbox("Usar una columna de Precio como regresor (Solo Prophet)", value=True)
            
            regressor_col = None
            if use_regressor:
                price_col = st.selectbox("Columna de Precio:", 
                                         options=['(Ninguno)'] + [c for c in column_names if c not in [ds_col, y_col]], 
                                         index=min(1, len(column_names) - 1))
                if price_col != '(Ninguno)':
                    regressor_col = 'precio_regressor' # Nombre temporal para el modelo
            
            # --- Preprocesamiento y Renombre ---
            try:
                # Renombrar las columnas al formato Prophet/Statsmodels
                df = df_original.rename(columns={ds_col: 'ds', y_col: 'y'})
                
                # Conversión de tipos de datos y manejo del Regresor
                df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                
                if regressor_col:
                    df = df.rename(columns={price_col: regressor_col})
                    df[regressor_col] = pd.to_numeric(df[regressor_col], errors='coerce')
                    df = df.dropna(subset=['ds', 'y', regressor_col])
                else:
                    df = df.dropna(subset=['ds', 'y'])
                
                # Asegurar que los datos estén ordenados y la frecuencia sea diaria
                df = df.sort_values('ds').reset_index(drop=True)
                
            except Exception as e:
                st.error(f"Error al asignar o convertir columnas. Detalle: {e}")
                df = None
            
            
            # --- Configuración del Pronóstico ---
            if df is not None and not df.empty:
                st.header("3. Configuración del Pronóstico")
                st.subheader("Horizonte")
                n_days = st.slider("Días a Pronosticar:", min_value=7, max_value=365, value=90, step=7)
                
                st.subheader("Parámetros de Statsmodels")
                st.markdown("Ajuste los coeficientes de suavizado (0: sin suavizado, 1: máximo suavizado).")

                # Parámetros para SES y Holt
                st.markdown("**SES / Holt ($\alpha$)** (Nivel de Suavizado)")
                alpha = st.slider("Smoothing Level ($\alpha$):", min_value=0.01, max_value=0.99, value=0.2, step=0.01)

                # Parámetros para Holt
                st.markdown("**Holt ($\beta$)** (Tendencia de Suavizado)")
                beta = st.slider("Smoothing Trend ($\beta$):", min_value=0.01, max_value=0.99, value=0.1, step=0.01)

                # Parámetros para Holt-Winters
                st.markdown("**Holt-Winters (Periodos y Tipo)**")
                seasonal_periods = st.selectbox("Periodos Estacionales (e.g., 7 para semanal):", options=[7, 30, 365], index=0)
                trend_type = st.selectbox("Tipo de Tendencia (Winter):", options=['add', 'mul'], index=0) # add o mul
                window_ma = st.slider("Ventana Promedio Móvil:", min_value=3, max_value=30, value=7, step=1)
                
# --- Contenido Principal ---
if df is not None and not df.empty:
    
    # Dividir datos: Usamos el 90% para entrenamiento y el 10% final para validación si hay suficientes datos
    split_point = int(len(df) * 0.90)
    df_train = df.iloc[:split_point]
    df_test = df.iloc[split_point:]
    
    st.header("Visualización y Modelado")
    
    # --- Ejecución y Pronóstico del Modelo ---
    if st.button("Generar y Comparar Pronósticos (6 Métodos)"):
        
        all_forecasts = {}
        mape_results = {}
        
        # 1. Prophet
        st.info("Calculando Prophet (con Precio como Regresor)...")
        prophet_forecast, prophet_mape = run_prophet(df_train, df, n_days, regressor_col)
        if isinstance(prophet_mape, str): st.error(prophet_mape)
        all_forecasts['Prophet'] = prophet_forecast
        mape_results['Prophet'] = prophet_mape
        
        # 2. Promedio Móvil (Moving Average)
        st.info("Calculando Promedio Móvil...")
        ma_forecast, ma_mape = run_statsmodels(df_train, n_days, 'Moving Average', window=window_ma)
        if isinstance(ma_mape, str): st.error(ma_mape)
        all_forecasts['Promedio Móvil'] = ma_forecast
        mape_results['Promedio Móvil'] = ma_mape
        
        # 3. Suavizado Exponencial Simple (SES)
        st.info("Calculando SES...")
        ses_forecast, ses_mape = run_statsmodels(df_train, n_days, 'SES', alpha=alpha)
        if isinstance(ses_mape, str): st.error(ses_mape)
        all_forecasts['SES'] = ses_forecast
        mape_results['SES'] = ses_mape

        # 4. Holt (Tendencia)
        st.info("Calculando Holt...")
        holt_forecast, holt_mape = run_statsmodels(df_train, n_days, 'Holt', alpha=alpha, beta=beta)
        if isinstance(holt_mape, str): st.error(holt_mape)
        all_forecasts['Holt'] = holt_forecast
        mape_results['Holt'] = holt_mape
        
        # 5. Holt-Winters Aditivo (Estacionalidad + Tendencia)
        st.info("Calculando Holt-Winters Aditivo...")
        winter_add_forecast, winter_add_mape = run_statsmodels(
            df_train, n_days, 'Winter_Add', seasonal_periods=seasonal_periods, trend_type=trend_type
        )
        if isinstance(winter_add_mape, str): st.error(winter_add_mape)
        all_forecasts['Winter Aditivo'] = winter_add_forecast
        mape_results['Winter Aditivo'] = winter_add_mape
        
        # 6. Holt-Winters Multiplicativo (Estacionalidad + Tendencia)
        st.info("Calculando Holt-Winters Multiplicativo...")
        winter_mul_forecast, winter_mul_mape = run_statsmodels(
            df_train, n_days, 'Winter_Mult', seasonal_periods=seasonal_periods, trend_type=trend_type
        )
        if isinstance(winter_mul_mape, str): st.error(winter_mul_mape)
        all_forecasts['Winter Multiplicativo'] = winter_mul_forecast
        mape_results['Winter Multiplicativo'] = winter_mul_mape
        
        st.success("¡Todos los pronósticos generados!")

        # --- TABLA RESUMEN DE ERRORES (MAPE%) ---
        
        st.header("5. Tabla Resumen de Error (%MAPE)")
        
        # Filtrar solo resultados válidos
        valid_results = {k: v for k, v in mape_results.items() if not isinstance(v, str)}
        
        df_mape = pd.DataFrame.from_dict(valid_results, orient='index', columns=['Error %MAPE'])
        df_mape['Error %MAPE'] = df_mape['Error %MAPE'].map('{:.2f}%'.format)
        df_mape = df_mape.sort_values(by='Error %MAPE', ascending=True)

        st.markdown("**Métrica usada: Mean Absolute Percentage Error (MAPE)**. *El valor más bajo es el mejor.*")
        st.dataframe(df_mape, use_container_width=True)

        # --- GRÁFICO DE COMPARACIÓN ---
        
        st.header("6. Comparación Gráfica de Pronósticos")
        
        fig = go.Figure()
        
        # 1. Trazar Datos Históricos (Actuales)
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Demanda Histórica (Real)', line=dict(color='black', width=2)))
        
        # 2. Trazar cada Pronóstico
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        i = 0
        for name, forecast_df in all_forecasts.items():
            if forecast_df is not None:
                # Obtener solo las fechas futuras
                future_dates = forecast_df[forecast_df['ds'] > df['ds'].max()]
                
                # Trazar solo el pronóstico futuro (yhat o el nombre original si es prophet)
                yhat_col = 'yhat' if 'yhat' in future_dates.columns else 'yhat'
                
                fig.add_trace(go.Scatter(
                    x=future_dates['ds'], 
                    y=future_dates[yhat_col], 
                    mode='lines', 
                    name=f'{name} ({mape_results[name]:.2f}%)',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
                i += 1
                
        fig.update_layout(
            title=f'Pronóstico de Demanda para los Próximos {n_days} Días',
            xaxis_title='Fecha',
            yaxis_title=f"Cantidad de Servicios ({y_col})",
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.divider()
    
else:
    st.info("Por favor, sube un archivo (CSV, XLSX o XLSM) en la barra lateral y asigna las columnas para comenzar.")
