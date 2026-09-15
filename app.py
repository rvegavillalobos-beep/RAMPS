import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Calculador de Perfil de Conveyor", layout="wide")

st.title("Calculador de Perfiles de Aceleración y Frenado - Conveyor")
st.markdown("Gráficas interactivas con Plotly para hacer zoom, pan y análisis detallado.")

# --- INICIALIZACIÓN DE ESTADOS ---
defaults = {
    "conveyor_length": 3000.0,
    "speed_fast_a": 300.0,
    "speed_slow_a": 100.0,
    "accel_a": 300.0,
    "decel_a": 300.0,
    "sensor_distance_a": 150.0,
    "ramp_stop_a": 150.0,  # <-- Nuevo parámetro en ms (SEW)
    
    "comparar": False,
    "speed_fast_b": 450.0,
    "speed_slow_b": 120.0,
    "accel_b": 400.0,
    "decel_b": 300.0,
    "sensor_distance_b": 150.0,
    "ramp_stop_b": 0.0     # <-- Nuevo parámetro B en ms
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- BARRA LATERAL DE PARÁMETROS ---
st.sidebar.header("Geometría Global")
st.sidebar.number_input("Largo total del conveyor (mm)", value=st.session_state.conveyor_length, step=100.0, key="conveyor_length")

st.sidebar.header("Perfil Principal (Perfil A)")
st.sidebar.number_input("SPEED_AUTO_FAST A (mm/s)", value=st.session_state.speed_fast_a, step=10.0, key="speed_fast_a")
st.sidebar.number_input("SPEED_AUTO_SLOW A (mm/s)", value=st.session_state.speed_slow_a, step=10.0, key="speed_slow_a")
st.sidebar.number_input("RAMP_ACCEL A (mm/s²)", value=st.session_state.accel_a, step=50.0, key="accel_a")
st.sidebar.number_input("RAMP_DECEL A (mm/s²)", value=st.session_state.decel_a, step=50.0, key="decel_a")
st.sidebar.number_input("Distancia Sensor Reducción A (mm)", value=st.session_state.sensor_distance_a, step=50.0, key="sensor_distance_a")
st.sidebar.number_input("RAMP_STOP A (ms)", value=st.session_state.ramp_stop_a, step=10.0, min_value=0.0, key="ramp_stop_a", help="Tiempo de rampa de parada en el drive SEW al tocar el sensor Stop")

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

# --- FUNCIÓN DE CÁLCULO CINEMÁTICO CORREGIDA ---
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist, ramp_stop_ms):
    dt = 0.001  # Paso de tiempo de 1 ms para alta precisión
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
    
    t_accel_end = 0.0
    t_sensor_red = 0.0
    t_sensor_stop = 0.0
    
    # Calcular aceleración de freno para RAMP_STOP (mm/s²)
    # Si RAMP_STOP > 0, desacelera desde v_slow hasta 0 en ramp_stop_ms
    if ramp_stop_ms > 0:
        a_stop = v_slow / (ramp_stop_ms / 1000.0)
    else:
        a_stop = 1e6  # Paro casi instantáneo por simulación
        
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
                state = "CRUISE_SLOW"
        elif state == "CRUISE_SLOW":
            # Avanza a v_slow hasta tocar físicamente el sensor de stop
            if p >= pos_stop:
                t_sensor_stop = t[i]
                if ramp_stop_ms <= 0:
                    v = 0.0
                    state = "DONE"
                else:
                    state = "DECEL_TO_STOP"
        elif state == "DECEL_TO_STOP":
            v -= a_stop * dt
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
            
    # Distancia de rebasamiento (overrun)
    dist_overrun = pos[-1] - pos_stop
    
    # Deceleración en Gs generada en el paro
    decel_stop_m_s2 = (a_stop if ramp_stop_ms > 0 else 0.1 / dt) / 1000.0
    g_force_stop = decel_stop_m_s2 / 9.81 if ramp_stop_ms > 0 else 0.0
    
    return t, pos, vel, t_sensor_red, t_sensor_stop, dist_overrun, g_force_stop

# Ejecución de cálculos
t_a, pos_a, vel_a, t_red_a, t_stop_a, overrun_a, g_a = calcular_perfil(
    st.session_state.speed_fast_a, st.session_state.speed_slow_a, 
    st.session_state.accel_a, st.session_state.decel_a, 
    st.session_state.conveyor_length, st.session_state.sensor_distance_a,
    st.session_state.ramp_stop_a
)

if st.session_state.comparar:
    t_b, pos_b, vel_b, t_red_b, t_stop_b, overrun_b, g_b = calcular_perfil(
        st.session_state.speed_fast_b, st.session_state.speed_slow_b, 
        st.session_state.accel_b, st.session_state.decel_b, 
        st.session_state.conveyor_length, st.session_state.sensor_distance_b,
        st.session_state.ramp_stop_b
    )

# --- CONSTRUCCIÓN DE GRÁFICAS CON PLOTLY ---
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Perfil de Velocidad", "Perfil de Posición"),
    horizontal_spacing=0.12
)

# --- GRÁFICA DE VELOCIDAD (Columna 1) ---
fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Velocidad A', line=dict(color='#1f77b4', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[t_red_a, t_red_a], y=[0, max(vel_a)*1.1], mode='lines', name='Sensor Reducción A', line=dict(color="#ff7f0e", width=2, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=[t_stop_a, t_stop_a], y=[0, max(vel_a)*1.1], mode='lines', name='Inicio Stop A', line=dict(color="#d62728", width=2, dash="dash")), row=1, col=1)

if st.session_state.comparar:
    max_v_gen = max(max(vel_a), max(vel_b))
    fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Velocidad B', line=dict(color='#9467bd', width=3, dash='dashdot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t_red_b, t_red_b], y=[0, max_v_gen*1.1], mode='lines', name='Sensor Reducción B', line=dict(color="#17becf", width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t_stop_b, t_stop_b], y=[0, max_v_gen*1.1], mode='lines', name='Inicio Stop B', line=dict(color="#e377c2", width=2, dash="dash")), row=1, col=1)

# --- GRÁFICA DE POSICIÓN (Columna 2) ---
fig.add_trace(go.Scatter(x=t_a, y=pos_a, mode='lines', name='Posición A', line=dict(color='#2ca02c', width=3)), row=1, col=2)
fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=st.session_state.conveyor_length, y1=st.session_state.conveyor_length, line=dict(color="#d62728", width=2, dash="dash"), row=1, col=2)
fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=st.session_state.conveyor_length-st.session_state.sensor_distance_a, y1=st.session_state.conveyor_length-st.session_state.sensor_distance_a, line=dict(color="#ff7f0e", width=2, dash="dot"), row=1, col=2)

if st.session_state.comparar:
    fig.add_trace(go.Scatter(x=t_b, y=pos_b, mode='lines', name='Posición B', line=dict(color='#8c564b', width=3, dash='dashdot')), row=1, col=2)
    fig.add_shape(type="line", x0=0, x1=t_b[-1], y0=st.session_state.conveyor_length-st.session_state.sensor_distance_b, y1=st.session_state.conveyor_length-st.session_state.sensor_distance_b, line=dict(color="#17becf", width=2, dash="dot"), row=1, col=2)

fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
fig.update_yaxes(title_text="Velocidad (mm/s)", row=1, col=1)
fig.update_xaxes(title_text="Tiempo (s)", row=1, col=2)
fig.update_yaxes(title_text="Posición (mm)", row=1, col=2)

fig.update_layout(
    height=550,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
    margin=dict(b=120)
)

st.plotly_chart(fig, use_container_width=True)

# --- MÉTRICAS DE PARO Y OVERRUN ---
st.markdown("---")
st.subheader("🎯 Análisis de Paro y Rebasamiento (Overrun)")

col_o1, col_o2, col_o3 = st.columns(3)
col_o1.metric("Distancia Rebasada (Overrun A)", f"{overrun_a:.2f} mm", help="Distancia recorrida tras la activación del sensor Stop")
col_o2.metric("Posición Final de Paro A", f"{pos_a[-1]:.2f} mm")
col_o3.metric("Aceleración de Impacto", f"{g_a:.3f} G" if st.session_state.ramp_stop_a > 0 else "Paro Seco / Inercial")

if st.session_state.comparar:
    st.markdown("**Comparativa de Rebasamiento con Perfil B:**")
    col_ob1, col_ob2, col_ob3 = st.columns(3)
    col_ob1.metric("Distancia Rebasada (Overrun B)", f"{overrun_b:.2f} mm")
    col_ob2.metric("Posición Final de Paro B", f"{pos_b[-1]:.2f} mm")
    col_ob3.metric("Aceleración de Impacto B", f"{g_b:.3f} G" if st.session_state.ramp_stop_b > 0 else "Paro Seco / Inercial")

# --- MÉTRICAS DE TIEMPO DEL CICLO ---
st.markdown("---")
st.subheader("⏱️ Desglose de Tiempos del Ciclo")

if not st.session_state.comparar:
    col_m0, col_m1, col_m2, col_m3 = st.columns(4)
    col_m0.metric("Tiempo Total Ciclo", f"{t_a[-1]:.2f} s")
    col_m1.metric("Tiempo a Sensor Reducción", f"{t_red_a:.2f} s")
    col_m2.metric("Tiempo a Sensor Stop", f"{t_stop_a:.2f} s")
    col_m3.metric("Tiempo en Rampa Stop", f"{(t_a[-1] - t_stop_a):.3f} s")
