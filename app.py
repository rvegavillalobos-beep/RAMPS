import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Calculador de Perfil de Conveyor", layout="wide")

st.title("Calculador de Perfiles de Aceleración y Frenado - Conveyor")
st.markdown("Gráficas interactivas con Plotly para hacer zoom, pan y análisis detallado de inercia.")

# --- INICIALIZACIÓN DE ESTADOS ---
defaults = {
    "conveyor_length": 3000.0,
    "speed_fast_a": 300.0,
    "speed_slow_a": 100.0,
    "accel_a": 300.0,
    "decel_a": 300.0,
    "sensor_distance_a": 150.0,
    "ramp_stop_a": 150.0,
    "mu_a": 0.28,
    
    "comparar": False,
    "speed_fast_b": 450.0,
    "speed_slow_b": 120.0,
    "accel_b": 400.0,
    "decel_b": 300.0,
    "sensor_distance_b": 150.0,
    "ramp_stop_b": 0.0,
    "mu_b": 0.28
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- BARRA LATERAL DE PARÁMETROS ---
st.sidebar.header("Geometría y Física Global")
st.sidebar.number_input("Largo total del conveyor (mm)", value=st.session_state.conveyor_length, step=100.0, key="conveyor_length")

st.sidebar.header("Perfil Principal (Perfil A)")
st.sidebar.number_input("SPEED_AUTO_FAST A (mm/s)", value=st.session_state.speed_fast_a, step=10.0, key="speed_fast_a")
st.sidebar.number_input("SPEED_AUTO_SLOW A (mm/s)", value=st.session_state.speed_slow_a, step=10.0, key="speed_slow_a")
st.sidebar.number_input("RAMP_ACCEL A (mm/s²)", value=st.session_state.accel_a, step=50.0, key="accel_a")
st.sidebar.number_input("RAMP_DECEL A (mm/s²)", value=st.session_state.decel_a, step=50.0, key="decel_a")
st.sidebar.number_input("Distancia Sensor Reducción A (mm)", value=st.session_state.sensor_distance_a, step=50.0, key="sensor_distance_a")
st.sidebar.number_input("RAMP_STOP A (ms)", value=st.session_state.ramp_stop_a, step=10.0, min_value=0.0, key="ramp_stop_a", help="Rampa SEW de parada en el sensor stop")
st.sidebar.number_input("Coeficiente Fricción μ A", value=st.session_state.mu_a, step=0.01, min_value=0.01, max_value=1.0, key="mu_a")

st.sidebar.markdown("---")
st.sidebar.checkbox("Comparar con Perfil B", value=st.session_state.comparar, key="comparar")

if st.session_state.comparar:
    st.sidebar.header("Perfil de Comparación (Perfil B)")
    st.sidebar.number_input("SPEED_AUTO_FAST B (mm/s)", value=st.session_state.speed_fast_b, step=10.0, key="speed_fast_b")
    st.sidebar.number_input("SPEED_AUTO_SLOW B (mm/s)", value=st.session_state.speed_slow_b, step=10.0, key="speed_slow_b")
    st.sidebar.number_input("RAMP_ACCEL B (mm/s²)", value=st.session_state.accel_b, step=50.0, key="accel_b")
    st.sidebar.number_input("RAMP_DECEL B (mm/s²)", value=st.session_state.decel_b, step=50.0, key="decel_b")
    st.sidebar.number_input("Distancia Sensor Reducción B (mm)", value=st.session_state.sensor_distance_b, step=50.0, key="sensor_distance_b")
    st.sidebar.number_input("RAMP_STOP B (ms)", value=st.session_state.ramp_stop_b, step=10.0, min_value=0.0, key="ramp_stop_b")
    st.sidebar.number_input("Coeficiente Fricción μ B", value=st.session_state.mu_b, step=0.01, min_value=0.01, max_value=1.0, key="mu_b")

# --- FUNCIÓN DE CÁLCULO CINEMÁTICO REALISTA ---
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist, ramp_stop_ms, mu):
    dt = 0.001  # Resolución de 1 ms
    t_max = 25.0
    steps = int(t_max / dt)
    
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)
    
    p = 0.0
    v = 0.0
    pos_sensor_red = length - s_dist
    pos_stop = length
    state = "ACCEL_FAST"
    
    t_sensor_red = 0.0
    t_sensor_stop = 0.0
    
    # 1. Filtro de elasticidad mecánica mínima (20 ms por flexión de chasis/cadena/juegos)
    T_MIN_MECANICO_MS = 20.0
    ramp_stop_real_ms = max(ramp_stop_ms, T_MIN_MECANICO_MS)
    
    # 2. Desaceleración real del conveyor (mm/s²)
    a_stop_conveyor = (v_slow / (ramp_stop_real_ms / 1000.0))
    g_conveyor = (a_stop_conveyor / 1000.0) / 9.81
    
    # 3. Límite de aceleración por fricción en la pieza (mm/s²)
    g_max_pieza = mu
    a_max_pieza = mu * 9.81 * 1000.0  # mm/s²
    
    for i in range(1, steps):
        t[i] = t[i-1] + dt
        
        if state == "ACCEL_FAST":
            v += accel * dt
            if v >= v_fast:
                v = v_fast
                state = "CRUISE_FAST"
        elif state == "CRUISE_FAST":
            if p >= pos_sensor_red:
                t_sensor_red = t[i]
                state = "DECEL_TO_SLOW"
        elif state == "DECEL_TO_SLOW":
            v -= decel * dt
            if v <= v_slow:
                v = v_slow
                state = "CRUISE_SLOW"
        elif state == "CRUISE_SLOW":
            if p >= pos_stop:
                t_sensor_stop = t[i]
                state = "DECEL_TO_STOP"
        elif state == "DECEL_TO_STOP":
            v -= a_stop_conveyor * dt
            if v <= 0:
                v = 0.0
                state = "DONE"
        elif state == "DONE":
            v = 0.0
            
        p += v * dt
        pos[i] = p
        vel[i] = v
        
        if state == "DONE" and i > 50 and np.all(vel[i-20:i] == 0):
            t = t[:i+1]
            pos = pos[:i+1]
            vel = vel[:i+1]
            break
            
    # Distancia recorrida por el conveyor tras la señal del sensor stop
    dist_overrun_conveyor = pos[-1] - pos_stop
    
    # 4. Cálculo de deslizamiento relativo de la pieza sobre la charola
    se_desliza = g_conveyor > g_max_pieza
    if se_desliza:
        # Distancia de frenado de la pieza impulsada por fricción
        dist_freno_pieza = (v_slow ** 2) / (2 * a_max_pieza)
        # Distancia de frenado del conveyor
        dist_freno_conveyor = (v_slow ** 2) / (2 * a_stop_conveyor)
        deslizamiento_mm = dist_freno_pieza - dist_freno_conveyor
        g_pieza_real = g_max_pieza
    else:
        deslizamiento_mm = 0.0
        g_pieza_real = g_conveyor
        
    return t, pos, vel, t_red_a if 't_red_a' in locals() else t_sensor_red, t_sensor_stop, dist_overrun_conveyor, g_conveyor, g_pieza_real, se_desliza, deslizamiento_mm

# Ejecución de cálculos
t_a, pos_a, vel_a, t_red_a, t_stop_a, overrun_a, g_conv_a, g_pieza_a, desliza_a, d_desliza_a = calcular_perfil(
    st.session_state.speed_fast_a, st.session_state.speed_slow_a, 
    st.session_state.accel_a, st.session_state.decel_a, 
    st.session_state.conveyor_length, st.session_state.sensor_distance_a,
    st.session_state.ramp_stop_a, st.session_state.mu_a
)

if st.session_state.comparar:
    t_b, pos_b, vel_b, t_red_b, t_stop_b, overrun_b, g_conv_b, g_pieza_b, desliza_b, d_desliza_b = calcular_perfil(
        st.session_state.speed_fast_b, st.session_state.speed_slow_b, 
        st.session_state.accel_b, st.session_state.decel_b, 
        st.session_state.conveyor_length, st.session_state.sensor_distance_b,
        st.session_state.ramp_stop_b, st.session_state.mu_b
    )

# --- GRÁFICAS CON PLOTLY ---
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Perfil de Velocidad", "Perfil de Posición del Conveyor"),
    horizontal_spacing=0.12
)

fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Velocidad A', line=dict(color='#1f77b4', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[t_red_a, t_red_a], y=[0, max(vel_a)*1.1], mode='lines', name='Sensor Reducción A', line=dict(color="#ff7f0e", width=2, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=[t_stop_a, t_stop_a], y=[0, max(vel_a)*1.1], mode='lines', name='Inicio Stop A', line=dict(color="#d62728", width=2, dash="dash")), row=1, col=1)

if st.session_state.comparar:
    max_v_gen = max(max(vel_a), max(vel_b))
    fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Velocidad B', line=dict(color='#9467bd', width=3, dash='dashdot')), row=1, col=1)

fig.add_trace(go.Scatter(x=t_a, y=pos_a, mode='lines', name='Posición A', line=dict(color='#2ca02c', width=3)), row=1, col=2)
fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=st.session_state.conveyor_length, y1=st.session_state.conveyor_length, line=dict(color="#d62728", width=2, dash="dash"), row=1, col=2)
fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=st.session_state.conveyor_length-st.session_state.sensor_distance_a, y1=st.session_state.conveyor_length-st.session_state.sensor_distance_a, line=dict(color="#ff7f0e", width=2, dash="dot"), row=1, col=2)

if st.session_state.comparar:
    fig.add_trace(go.Scatter(x=t_b, y=pos_b, mode='lines', name='Posición B', line=dict(color='#8c564b', width=3, dash='dashdot')), row=1, col=2)

fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
fig.update_yaxes(title_text="Velocidad (mm/s)", row=1, col=1)
fig.update_xaxes(title_text="Tiempo (s)", row=1, col=2)
fig.update_yaxes(title_text="Posición (mm)", row=1, col=2)

fig.update_layout(
    height=500, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
    margin=dict(b=120)
)

st.plotly_chart(fig, use_container_width=True)

# --- MÉTRICAS DE ESTABILIDAD Y DESLIZAMIENTO ---
st.markdown("---")
st.subheader("⚠️ Análisis de Estabilidad y Deslizamiento de la Carga")

col_e1, col_e2, col_e3, col_e4 = st.columns(4)
col_e1.metric("Aceleración Conveyor", f"{g_conv_a:.3f} G")
col_e2.metric("Límite Fricción (μ)", f"{st.session_state.mu_a:.2f} G")
col_e3.metric("Estado de la Carga", "🔴 SE DESLIZA" if desliza_a else "🟢 ESTABLE")
col_e4.metric("Deslizamiento Relativo", f"{d_desliza_a:.2f} mm" if desliza_a else "0.00 mm")

# --- MÉTRICAS DE PARO Y POSICIONAMIENTO ---
st.markdown("---")
st.subheader("🎯 Posicionamiento y Tiempos (Perfil A)")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p1.metric("Rebasamiento Conveyor", f"{overrun_a:.2f} mm")
col_p2.metric("Posición Final Conveyor", f"{pos_a[-1]:.2f} mm")
col_p3.metric("Posición Final Pieza", f"{(pos_a[-1] + d_desliza_a):.2f} mm")
col_p4.metric("Tiempo Total Ciclo", f"{t_a[-1]:.2f} s")
