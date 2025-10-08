import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go
import datetime
import io
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Comparador de Pronóstico de Demanda",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Demand Forecasting Dashboard 📈📊")
st.markdown("Compara 6 métodos de pronóstico, incluyendo Prophet con regresor de precio y 5 modelos de Suavizado Exponencial, con múltiples métricas de error.")

# --- Funciones de Carga y Métrica ---

@st.cache_data
def load_data(file):
    """Carga el archivo de Excel (.xlsx, .xlsm) o CSV."""
    
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

def calculate_mape(actual, predicted):
    """Calcula el Mean Absolute Percentage Error (MAPE)."""
    temp_df = pd.DataFrame({'actual': actual, 'predicted': predicted}).dropna()
    actual = temp_df['actual'].values
    predicted = temp_df['predicted'].values
    
    if len(actual) == 0:
        return np.nan
    
    # Evitar división por cero
    actual[actual == 0] = 1e-8 
    
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mape

def calculate_mad_mae(actual, predicted):
    """Calcula el Mean Absolute Deviation (MAD) o Mean Absolute Error (MAE)."""
    temp_df = pd.DataFrame({'actual': actual, 'predicted': predicted}).dropna()
    actual = temp_df['actual'].values
    predicted = temp_df['predicted'].values
    
    if len(actual) == 0:
        return np.nan
        
    mad = np.mean(np.abs(actual - predicted))
    return mad

def calculate_rmse(actual, predicted):
    """Calcula el Root Mean Squared Error (RMSE)."""
    temp_df = pd.DataFrame({'actual': actual, 'predicted': predicted}).dropna()
    actual = temp_df['actual'].values
    predicted = temp_df['predicted'].values
    
    if len(actual) == 0:
        return np.nan
        
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    return rmse

def calculate_tracking_signal(actual, predicted):
    """Calcula el Tracking Signal (Señal de Rastreo)."""
    
    temp_df = pd.DataFrame({'actual': actual, 'predicted': predicted}).dropna()
    
    if len(temp_df) == 0:
        return np.nan
        
    # Error de Pronóstico (FE)
    errors = temp_df['actual'] - temp_df['predicted']
    
    # Suma Corriente de Errores de Pronóstico (RSFE)
    rsfe = errors.sum()
    
    # Desviación Media Absoluta (MAD)
    # Reutilizamos la función MAD, que es el promedio de los errores absolutos.
    mad = calculate_mad_mae(actual, predicted)
    
    # Calcular Tracking Signal (TS = RSFE / MAD)
    if mad == 0:
        # Evitar división por cero. Si MAD es 0, el error es nulo.
        return 0.0
    
    ts = rsfe / mad
    return ts

# --- Funciones de Modelado (Prophet y Statsmodels) ---

def run_prophet(df_train, df_full, n_days, regressor_name):
    """Ejecuta el modelo Prophet y devuelve todas las métricas de error."""
    try:
        df_prophet = df_train[['ds', 'y']].copy()
        
        m = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
        if regressor_name:
            df_prophet[regressor_name] = df_train[regressor_name]
            m.add_regressor(regressor_name)
        
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=n_days)
        
        if regressor_name:
            future[regressor_name] = df_full[regressor_name].mean()
            
        forecast = m.predict(future)
        
        # Unir la predicción con los datos históricos para calcular el error
        df_join = df_prophet.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        
        # CÁLCULO DE MÉTRICAS
        actual = df_join['y']
        predicted = df_join['yhat']
        
        metrics = {
            'MAPE': calculate_mape(actual, predicted),
            'MAD (MAE)': calculate_mad_mae(actual, predicted),
            'RMSE': calculate_rmse(actual, predicted),
            'Tracking Signal': calculate_tracking_signal(actual, predicted)
        }
        
        return forecast, metrics
    except Exception as e:
        return None, {'MAPE': np.nan, 'MAD (MAE)': np.nan, 'RMSE': np.nan, 'Tracking Signal': f"Error: {e}"}

def run_statsmodels(df_train, n_days, model_type, **kwargs):
    """Ejecuta modelos de suavizado exponencial y devuelve todas las métricas de error."""
    
    metrics = {'MAPE': np.nan, 'MAD (MAE)': np.nan, 'RMSE': np.nan, 'Tracking Signal': np.nan}
    forecast_df = None
    
    try:
        y_train = df_train.set_index('ds')['y']
        
        if model_type == 'Moving Average':
            window = kwargs.get('window', 7)
            
            fit = y_train.rolling(window=window).mean()
            
            # CÁLCULO DE MÉTRICAS (en la porción predicha del entrenamiento)
            actual = y_train[window:]
            predicted = fit[window:]
            
            # Predicción futura
            last_ma_value = fit.iloc[-1] if not fit.empty else y_train.mean()
            forecast_index = pd.date_range(start=df_train['ds'].max() + pd.Timedelta(days=1), periods=n_days, freq='D')
            forecast_values = np.full(n_days, last_ma_value)
            forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast_values})
            
        else: # SES, Holt, Winter
            if model_type.startswith('Winter'):
                seasonal_periods = kwargs.get('seasonal_periods', 7)
                if len(y_train) < 2 * seasonal_periods:
                     raise ValueError(f"Faltan datos. Se necesitan al menos {2 * seasonal_periods} días para estacionalidad.")
            
            if model_type == 'SES':
                fit_model = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(
                    smoothing_level=kwargs.get('alpha'), optimized=False
                )
            elif model_type == 'Holt':
                fit_model = Holt(y_train, initialization_method="estimated").fit(
                    smoothing_level=kwargs.get('alpha'), smoothing_trend=kwargs.get('beta'), optimized=False
                )
            elif model_type == 'Winter_Add' or model_type == 'Winter_Mult':
                seasonal_type = 'add' if model_type == 'Winter_Add' else 'mul'
                fit_model = ExponentialSmoothing(
                    y_train, seasonal_periods=seasonal_periods, trend=kwargs.get('trend_type'), seasonal=seasonal_type, 
                    initialization_method="estimated"
                ).fit(
                    smoothing_level=kwargs.get('alpha'), smoothing_trend=kwargs.get('beta'), 
                    smoothing_seasonal=kwargs.get('gamma'), optimized=False
                )
            
            # CÁLCULO DE MÉTRICAS
            actual = y_train.values
            predicted = fit_model.fittedvalues.values
            
            # Predicción futura
            forecast = fit_model.predict(start=len(y_train), end=len(y_train) + n_days - 1)
            forecast_df = pd.DataFrame({'ds': forecast.index, 'yhat': forecast.values})

        # Aplicar Métricas después del cálculo del modelo
        metrics['MAPE'] = calculate_mape(actual, predicted)
        metrics['MAD (MAE)'] = calculate_mad_mae(actual, predicted)
        metrics['RMSE'] = calculate_rmse(actual, predicted)
        metrics['Tracking Signal'] = calculate_tracking_signal(actual, predicted)
        
        # Aseguramos que 'ds' esté en formato datetime si la indexación lo modificó
        if forecast_df is not None and 'ds' not in forecast_df.columns:
            forecast_df['ds'] = forecast_df.index
            
        return forecast_df, metrics
        
    except ValueError as ve:
        return None, {'MAPE': np.nan, 'MAD (MAE)': np.nan, 'RMSE': np.nan, 'Tracking Signal': f"Error: {ve}"}
    except Exception as e:
        return None, {'MAPE': np.nan, 'MAD (MAE)': np.nan, 'RMSE': np.nan, 'Tracking Signal': f"Error: {e}"}

# --- FUNCIÓN PARA EXCEL ---

@st.cache_data
def to_excel_buffer(all_forecasts, df_historical_max_date, y_col_name, all_metrics):
    """
    Crea un archivo Excel en memoria (buffer) con una hoja por cada pronóstico.
    """
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd')
    
    for name, forecast_df in all_forecasts.items():
        if forecast_df is not None:
            future_data = forecast_df[forecast_df['ds'] > df_historical_max_date].copy()
            
            # Formatear el DataFrame para la hoja de Excel
            data_to_save = future_data[['ds', 'yhat']].copy()
            data_to_save.columns = ['Fecha', f'Demanda_Estimada ({y_col_name})']
            
            # Incluir intervalos de confianza si existen
            if name == 'Prophet' and 'yhat_lower' in future_data.columns:
                 data_to_save['Límite_Inferior'] = future_data['yhat_lower']
                 data_to_save['Límite_Superior'] = future_data['yhat_upper']

            # Añadir la información de las métricas como primera fila
            metrics_dict = all_metrics.get(name, {})
            
            # Crear la fila de resumen de métricas
            rows_list = []
            for metric, value in metrics_dict.items():
                if isinstance(value, (float, np.float64)) and not np.isnan(value):
                    formatted_value = f"{value:.2f}"
                    if metric == 'MAPE':
                        formatted_value += '%'
                    
                    rows_list.append(
                        pd.DataFrame({'Fecha': [f'{metric}: {formatted_value}'], 
                                      'Demanda_Estimada ({y_col_name})': [''], 
                                      'Límite_Inferior': [''], 
                                      'Límite_Superior': ['']})
                    )
                elif isinstance(value, str) and value.startswith("Error"):
                     rows_list.append(
                        pd.DataFrame({'Fecha': [f'{metric}: {value}'], 
                                      'Demanda_Estimada ({y_col_name})': [''], 
                                      'Límite_Inferior': [''], 
                                      'Límite_Superior': ['']})
                    )

            if rows_list:
                metrics_info = pd.concat(rows_list)
                data_to_save = pd.concat([metrics_info, data_to_save], ignore_index=True)
            
            # Escribir la hoja
            sheet_name = name.replace(' ', '_').replace('-', '_')
            data_to_save.to_excel(writer, sheet_name=sheet_name[:31], index=False, startrow=0)
            
    writer.close()
    processed_data = output.getvalue()
    return processed_data

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
                st.header("3. Configuración de Modelos")
                st.subheader("Horizonte")
                n_days = st.slider("Días a Pronosticar:", min_value=7, max_value=365, value=90, step=7)
                
                st.subheader("Parámetros de Suavizado (α, β, γ)")
                st.markdown("Ajuste los coeficientes: $\\alpha$ (Nivel), $\\beta$ (Tendencia), $\\gamma$ (Estacionalidad).")

                # Parámetros para SES, Holt y Winter
                alpha = st.slider("Smoothing Level (α):", min_value=0.01, max_value=0.99, value=0.2, step=0.01)

                # Parámetros para Holt y Winter
                beta = st.slider("Smoothing Trend (β):", min_value=0.01, max_value=0.99, value=0.1, step=0.01)

                # Parámetros para Holt-Winters
                gamma = st.slider("Smoothing Seasonal (γ):", min_value=0.01, max_value=0.99, value=0.1, step=0.01)
                
                st.subheader("Opciones Específicas")
                
                # Opciones para Holt-Winters
                seasonal_periods = st.selectbox("Periodos Estacionales (e.g., 7 para semanal):", options=[7, 30, 365], index=0)
                trend_type = st.selectbox("Tipo de Tendencia (Winter):", options=['add', 'mul'], index=0) # add o mul
                
                # Opciones para Promedio Móvil
                window_ma = st.slider("Ventana Promedio Móvil:", min_value=3, max_value=30, value=7, step=1)
                
# --- Contenido Principal ---
if df is not None and not df.empty:
    
    # Dividir datos: Usamos el 90% para entrenamiento
    split_point = int(len(df) * 0.90)
    df_train = df.iloc[:split_point]
    
    st.header("Visualización y Modelado")
    
    # --- Ejecución y Pronóstico del Modelo ---
    if st.button("Generar y Comparar Pronósticos (6 Métodos)"):
        
        all_forecasts = {}
        all_metrics = {}
        
        # 1. Prophet
        st.info("Calculando Prophet (con Precio como Regresor)...")
        prophet_forecast, prophet_metrics = run_prophet(df_train, df, n_days, regressor_col)
        all_forecasts['Prophet'] = prophet_forecast
        all_metrics['Prophet'] = prophet_metrics
        if isinstance(prophet_metrics.get('Tracking Signal'), str): st.error(prophet_metrics['Tracking Signal'])
        
        # 2. Promedio Móvil
        st.info("Calculando Promedio Móvil...")
        ma_forecast, ma_metrics = run_statsmodels(df_train, n_days, 'Moving Average', window=window_ma)
        all_forecasts['Promedio Móvil'] = ma_forecast
        all_metrics['Promedio Móvil'] = ma_metrics
        if isinstance(ma_metrics.get('Tracking Signal'), str): st.error(ma_metrics['Tracking Signal'])
        
        # 3. Suavizado Exponencial Simple (SES)
        st.info("Calculando SES...")
        ses_forecast, ses_metrics = run_statsmodels(df_train, n_days, 'SES', alpha=alpha)
        all_forecasts['SES'] = ses_forecast
        all_metrics['SES'] = ses_metrics
        if isinstance(ses_metrics.get('Tracking Signal'), str): st.error(ses_metrics['Tracking Signal'])

        # 4. Holt (Tendencia)
        st.info("Calculando Holt...")
        holt_forecast, holt_metrics = run_statsmodels(df_train, n_days, 'Holt', alpha=alpha, beta=beta)
        all_forecasts['Holt'] = holt_forecast
        all_metrics['Holt'] = holt_metrics
        if isinstance(holt_metrics.get('Tracking Signal'), str): st.error(holt_metrics['Tracking Signal'])
        
        # 5. Holt-Winters Aditivo
        st.info("Calculando Holt-Winters Aditivo...")
        winter_add_forecast, winter_add_metrics = run_statsmodels(
            df_train, n_days, 'Winter_Add', 
            seasonal_periods=seasonal_periods, trend_type=trend_type, 
            alpha=alpha, beta=beta, gamma=gamma
        )
        all_forecasts['Winter Aditivo'] = winter_add_forecast
        all_metrics['Winter Aditivo'] = winter_add_metrics
        if isinstance(winter_add_metrics.get('Tracking Signal'), str) and "Faltan datos" in winter_add_metrics['Tracking Signal']: 
            st.warning(f"⚠️ **Error en Winter Aditivo:** {winter_add_metrics['Tracking Signal']} Por favor, ajusta los Periodos Estacionales o usa más datos de entrenamiento.")
        elif isinstance(winter_add_metrics.get('Tracking Signal'), str): 
            st.error(winter_add_metrics['Tracking Signal'])
            
        # 6. Holt-Winters Multiplicativo
        st.info("Calculando Holt-Winters Multiplicativo...")
        winter_mul_forecast, winter_mul_metrics = run_statsmodels(
            df_train, n_days, 'Winter_Mult', 
            seasonal_periods=seasonal_periods, trend_type=trend_type, 
            alpha=alpha, beta=beta, gamma=gamma
        )
        all_forecasts['Winter Multiplicativo'] = winter_mul_forecast
        all_metrics['Winter Multiplicativo'] = winter_mul_metrics
        if isinstance(winter_mul_metrics.get('Tracking Signal'), str) and "Faltan datos" in winter_mul_metrics['Tracking Signal']: 
            st.warning(f"⚠️ **Error en Winter Multiplicativo:** {winter_mul_metrics['Tracking Signal']} Por favor, ajusta los Periodos Estacionales o usa más datos de entrenamiento.")
        elif isinstance(winter_mul_metrics.get('Tracking Signal'), str): 
            st.error(winter_mul_metrics['Tracking Signal'])
            
        st.success("¡Todos los pronósticos generados!")
        
        # --- TABLA RESUMEN DE ERRORES ---
        
        st.header("5. Tabla Resumen de Error (Métricas)")
        
        # Crear DataFrame de Métricas
        data = {model: {k: v for k, v in metrics.items() if not isinstance(v, str)} 
                for model, metrics in all_metrics.items()}
        df_metrics = pd.DataFrame.from_dict(data, orient='index')
        
        # Formatear y añadir descripciones
        df_metrics['Error %MAPE'] = df_metrics['MAPE'].map('{:.2f}%'.format)
        df_metrics['MAD (MAE)'] = df_metrics['MAD (MAE)'].map('{:.2f}'.format)
        df_metrics['RMSE'] = df_metrics['RMSE'].map('{:.2f}'.format)
        df_metrics['Tracking Signal (Sesgo)'] = df_metrics['Tracking Signal'].map('{:.2f}'.format)
        
        # Seleccionar y reordenar las columnas a mostrar
        df_metrics = df_metrics[['Error %MAPE', 'MAD (MAE)', 'RMSE', 'Tracking Signal (Sesgo)']]

        st.markdown(
            """
            - **MAPE (%):** Error porcentual promedio. **Más bajo es mejor.**
            - **MAD (MAE):** Error promedio en las unidades de la demanda. **Más bajo es mejor.**
            - **RMSE:** Error que penaliza más los errores grandes. En unidades de la demanda. **Más bajo es mejor.**
            - **Tracking Signal (TS):** Mide el sesgo. Valores entre -4 y +4 son aceptables. **TS > 4 indica subestimación; TS < -4 indica sobreestimación.**
            """
        )
        st.dataframe(df_metrics, use_container_width=True)

        # --- GRÁFICO DE COMPARACIÓN (Se mantiene sin cambios) ---
        
        st.header("6. Comparación Gráfica de Pronósticos")
        
        fig = go.Figure()
        
        # 1. Trazar Datos Históricos (Actuales)
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Demanda Histórica (Real)', line=dict(color='black', width=2)))
        
        # 2. Trazar cada Pronóstico
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        i = 0
        for name, forecast_df in all_forecasts.items():
            if forecast_df is not None:
                future_dates = forecast_df[forecast_df['ds'] > df['ds'].max()]
                
                mape_value = all_metrics.get(name, {}).get('MAPE')
                mape_label = f' ({mape_value:.2f}%)' if not np.isnan(mape_value) else ' (N/A)'
                
                fig.add_trace(go.Scatter(
                    x=future_dates['ds'], 
                    y=future_dates['yhat'], 
                    mode='lines', 
                    name=f'{name}{mape_label}',
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

        # --- BOTÓN DE DESCARGA DE EXCEL ---
        st.header("7. Descarga de Resultados")
        
        # Generar el archivo Excel en un buffer
        # ¡IMPORTANTE!: Se pasa 'all_metrics' en lugar de 'mape_results'
        excel_data = to_excel_buffer(all_forecasts, df['ds'].max(), y_col, all_metrics) 
        
        st.download_button(
            label="Descargar Pronósticos (Excel 6 Hojas) 📥",
            data=excel_data,
            file_name=f'pronostico_comparativo_{datetime.date.today()}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            help='Descarga un archivo Excel con una hoja para el pronóstico futuro de cada método.'
        )
        
    st.divider()
    
else:
    st.info("Por favor, sube un archivo y asigna las columnas en la barra lateral para comenzar. Asegúrate de tener al menos dos ciclos estacionales (ej. 14 días si usas estacionalidad semanal) para los modelos de Holt-Winters.")
