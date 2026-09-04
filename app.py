import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Calculador de Perfil de Conveyor", layout="wide")

st.title("Calculador de Perfiles de Aceleración y Frenado - Conveyor")
st.markdown("Gráficas interactivas con Plotly para hacer zoom, pan y análisis detallado.")

# --- BARRA LATERAL DE PARÁMETROS ---
st.sidebar.header("Geometría Global")
conveyor_length = st.sidebar.number_input("Largo total del conveyor (mm)", value=3000.0, step=100.0)

st.sidebar.header("Perfil Principal (Perfil A)")
speed_fast_a = st.sidebar.number_input("SPEED_AUTO_FAST A (mm/s)", value=300.0, step=10.0)
speed_slow_a = st.sidebar.number_input("SPEED_AUTO_SLOW A (mm/s)", value=100.0, step=10.0)
accel_a = st.sidebar.number_input("RAMP_ACCEL A (mm/s²)", value=600.0, step=50.0)
decel_a = st.sidebar.number_input("RAMP_DECEL A (mm/s²)", value=600.0, step=50.0)
sensor_distance_a = st.sidebar.number_input("Distancia Sensor Reducción A (mm)", value=800.0, step=50.0)

st.sidebar.markdown("---")
comparar = st.sidebar.checkbox("Comparar con Perfil B", value=False)

if comparar:
    st.sidebar.header("Perfil de Comparación (Perfil B)")
    speed_fast_b = st.sidebar.number_input("SPEED_AUTO_FAST B (mm/s)", value=450.0, step=10.0)
    speed_slow_b = st.sidebar.number_input("SPEED_AUTO_SLOW B (mm/s)", value=120.0, step=10.0)
    accel_b = st.sidebar.number_input("RAMP_ACCEL B (mm/s²)", value=800.0, step=50.0)
    decel_b = st.sidebar.number_input("RAMP_DECEL B (mm/s²)", value=400.0, step=50.0)
    sensor_distance_b = st.sidebar.number_input("Distancia Sensor Reducción B (mm)", value=1000.0, step=50.0)

# --- FUNCIÓN DE CÁLCULO CINEMÁTICO ---
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist):
    dt = 0.01
    t_max = 20.0
    steps = int(t_max / dt)
    
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)
    
    p = 0.0
    v = 0.0
    pos_sensor_red = length - s_dist
    pos_stop = length
    state = "ACCEL_FAST"
    
    t_accel_end = 0.0
    t_sensor_red = 0.0
    t_slow_reached = 0.0
    t_brake_start = 0.0
    
    for i in range(1, steps):
        t[i] = t[i-1] + dt
        
        if state == "ACCEL_FAST":
            v += accel * dt
            if v >= v_fast:
                v = v_fast
                t_accel_end = t[i]
                state = "CRUISE_FAST"
        elif state == "CRUISE_FAST":
            if p >= pos_sensor_red:
                t_sensor_red = t[i]
                state = "DECEL_TO_SLOW"
        elif state == "DECEL_TO_SLOW":
            v -= decel * dt
            if v <= v_slow:
                v = v_slow
                t_slow_reached = t[i]
                state = "CRUISE_SLOW"
        elif state == "CRUISE_SLOW":
            dist_frenado_nec = (v_slow**2) / (2 * decel) if decel > 0 else 0
            if p >= (pos_stop - dist_frenado_nec):
                t_brake_start = t[i]
                state = "DECEL_TO_STOP"
        elif state == "DECEL_TO_STOP":
            v -= decel * dt
            if v <= 0:
                v = 0.0
                state = "DONE"
        elif state == "DONE":
            v = 0.0
            
        p += v * dt
        if p > length:
            p = length
            v = 0.0
            
        pos[i] = p
        vel[i] = v
        
        if state == "DONE" and i > 50 and np.all(vel[i-20:i] == 0):
            t = t[:i+1]
            pos = pos[:i+1]
            vel = vel[:i+1]
            break
            
    return t, pos, vel, t_sensor_red, t_brake_start

t_a, pos_a, vel_a, t_red_a, t_stop_a = calcular_perfil(speed_fast_a, speed_slow_a, accel_a, decel_a, conveyor_length, sensor_distance_a)
if comparar:
    t_b, pos_b, vel_b, t_red_b, t_stop_b = calcular_perfil(speed_fast_b, speed_slow_b, accel_b, decel_b, conveyor_length, sensor_distance_b)

# --- CONSTRUCCIÓN DE GRÁFICAS CON PLOTLY ---
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Perfil de Velocidad", "Perfil de Posición"),
    horizontal_spacing=0.12
)

# --- GRÁFICA DE VELOCIDAD (Columna 1) ---
fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Velocidad A', line=dict(color='#1f77b4', width=3)), row=1, col=1)
fig.add_shape(type="line", x0=t_red_a, x1=t_red_a, y0=0, y1=max(vel_a)*1.1, line=dict(color="#ff7f0e", width=2, dash="dot"), row=1, col=1)
fig.add_shape(type="line", x0=t_stop_a, x1=t_stop_a, y0=0, y1=max(vel_a)*1.1, line=dict(color="#d62728", width=2, dash="dash"), row=1, col=1)

if comparar:
    fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Velocidad B', line=dict(color='#9467bd', width=3, dash='dashdot')), row=1, col=1)
    fig.add_shape(type="line", x0=t_red_b, x1=t_red_b, y0=0, y1=max(max(vel_a), max(vel_b))*1.1, line=dict(color="#17becf", width=2, dash="dot"), row=1, col=1)
    fig.add_shape(type="line", x0=t_stop_b, x1=t_stop_b, y0=0, y1=max(max(vel_a), max(vel_b))*1.1, line=dict(color="#e377c2", width=2, dash="dash"), row=1, col=1)

# --- GRÁFICA DE POSICIÓN (Columna 2) ---
fig.add_trace(go.Scatter(x=t_a, y=pos_a, mode='lines', name='Posición A', line=dict(color='#2ca02c', width=3)), row=1, col=2)
fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=conveyor_length, y1=conveyor_length, line=dict(color="#d62728", width=2, dash="dash"), row=1, col=2)
fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=conveyor_length-sensor_distance_a, y1=conveyor_length-sensor_distance_a, line=dict(color="#ff7f0e", width=2, dash="dot"), row=1, col=2)
fig.add_shape(type="line", x0=t_red_a, x1=t_red_a, y0=0, y1=conveyor_length, line=dict(color="rgba(255, 127, 14, 0.5)", width=1.5, dash="dot"), row=1, col=2)
fig.add_shape(type="line", x0=t_stop_a, x1=t_stop_a, y0=0, y1=conveyor_length, line=dict(color="rgba(214, 39, 40, 0.5)", width=1.5, dash="dash"), row=1, col=2)

if comparar:
    fig.add_trace(go.Scatter(x=t_b, y=pos_b, mode='lines', name='Posición B', line=dict(color='#8c564b', width=3, dash='dashdot')), row=1, col=2)
    fig.add_shape(type="line", x0=0, x1=t_b[-1], y0=conveyor_length-sensor_distance_b, y1=conveyor_length-sensor_distance_b, line=dict(color="#17becf", width=2, dash="dot"), row=1, col=2)
    fig.add_shape(type="line", x0=t_red_b, x1=t_red_b, y0=0, y1=conveyor_length, line=dict(color="rgba(23, 190, 207, 0.5)", width=1.5, dash="dot"), row=1, col=2)
    fig.add_shape(type="line", x0=t_stop_b, x1=t_stop_b, y0=0, y1=conveyor_length, line=dict(color="rgba(227, 119, 194, 0.5)", width=1.5, dash="dash"), row=1, col=2)

# Configuración general de ejes y layout interactivo nativo de Plotly
fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
fig.update_yaxes(title_text="Velocidad (mm/s)", row=1, col=1)
fig.update_xaxes(title_text="Tiempo (s)", row=1, col=2)
fig.update_yaxes(title_text="Posición (mm)", row=1, col=2)

fig.update_layout(
    height=550,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- MÉTRICAS COMPARATIVAS ---
st.markdown("---")
st.subheader("⏱️ Desglose de Tiempos del Ciclo")

if not comparar:
    col_m0, col_m1, col_m2, col_m3, col_m4 = st.columns(5)
    col_m0.metric("Tiempo Total Ciclo", f"{t_a[-1]:.2f} s")
    col_m1.metric("Aceleración", f"{t_red_a:.2f} s")
    col_m2.metric("Desacel. a Slow", f"{(t_stop_a - t_red_a):.2f} s")
    col_m3.metric("Velocidad Slow", f"{(t_a[-1] - t_stop_a):.2f} s")
    col_m4.metric("Frenado Final", f"{(t_a[-1] - t_stop_a):.2f} s")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Perfil A**")
        ca1, ca2, ca3 = st.columns(3)
        ca1.metric("Total Ciclo A", f"{t_a[-1]:.2f} s")
        ca2.metric("Sensor A", f"{sensor_distance_a} mm")
        ca3.metric("Frenado A", f"{decel_a} mm/s²")
    with col_b:
        st.markdown("**Perfil B**")
        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Total Ciclo B", f"{t_b[-1]:.2f} s")
        cb2.metric("Sensor B", f"{sensor_distance_b} mm")
        cb3.metric("Frenado B", f"{decel_b} mm/s²")
