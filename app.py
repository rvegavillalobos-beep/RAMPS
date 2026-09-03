import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calculador de Perfil de Conveyor", layout="wide")

st.title("Calculador de Perfiles de Aceleración y Frenado - Conveyor")
st.markdown("Ajusta los parámetros del sistema para generar el perfil cinemático estático.")

# --- BARRA LATERAL DE PARÁMETROS ---
st.sidebar.header("Parámetros del Sistema")
speed_fast = st.sidebar.number_input("SPEED_AUTO_FAST (mm/s)", value=300.0, step=10.0)
speed_slow = st.sidebar.number_input("SPEED_AUTO_SLOW (mm/s)", value=100.0, step=10.0)
accel = st.sidebar.number_input("RAMP_ACCELERATION (mm/s²)", value=600.0, step=50.0)
decel = st.sidebar.number_input("RAMP_DECELERATION (mm/s²)", value=600.0, step=50.0)

st.sidebar.header("Geometría del Conveyor")
conveyor_length = st.sidebar.number_input("Largo total del conveyor (mm)", value=3000.0, step=100.0)
sensor_distance = st.sidebar.number_input("Distancia Sensor Reducción a Stop (mm)", value=800.0, step=50.0)

# --- CÁLCULO FÍSICO DEL PERFIL ---
dt = 0.01
t_max = 20.0
steps = int(t_max / dt)

t = np.zeros(steps)
pos = np.zeros(steps)
vel = np.zeros(steps)
acc = np.zeros(steps)

p = 0.0
v = 0.0
pos_sensor_red = conveyor_length - sensor_distance
pos_stop = conveyor_length
state = "ACCEL_FAST"

for i in range(1, steps):
    t[i] = t[i-1] + dt
    
    if state == "ACCEL_FAST":
        a = accel
        v += a * dt
        if v >= speed_fast:
            v = speed_fast
            a = 0.0
            state = "CRUISE_FAST"
            
    elif state == "CRUISE_FAST":
        a = 0.0
        if p >= pos_sensor_red:
            state = "DECEL_TO_SLOW"
            
    elif state == "DECEL_TO_SLOW":
        a = -decel
        v += a * dt
        if v <= speed_slow:
            v = speed_slow
            a = 0.0
            state = "CRUISE_SLOW"
            
    elif state == "CRUISE_SLOW":
        a = 0.0
        # Distancia de frenado necesaria desde slow hasta 0
        dist_frenado_nec = (speed_slow**2) / (2 * decel) if decel > 0 else 0
        if p >= (pos_stop - dist_frenado_nec):
            state = "DECEL_TO_STOP"
            
    elif state == "DECEL_TO_STOP":
        a = -decel
        v += a * dt
        if v <= 0:
            v = 0.0
            a = 0.0
            state = "DONE"
            
    elif state == "DONE":
        a = 0.0
        v = 0.0
        
    p += v * dt
    if p > conveyor_length:
        p = conveyor_length
        v = 0.0
        a = 0.0
        
    pos[i] = p
    vel[i] = v
    acc[i] = a
    
    if state == "DONE" and i > 50 and np.all(vel[i-20:i] == 0):
        t = t[:i+1]
        pos = pos[:i+1]
        vel = vel[:i+1]
        acc = acc[:i+1]
        break

# --- VISUALIZACIÓN EN GRÁFICAS LIMPIAS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Perfil de Velocidad")
    fig_v, ax_v = plt.subplots(figsize=(6, 4))
    ax_v.plot(t, vel, color="tab:blue", linewidth=2.5)
    ax_v.set_xlabel("Tiempo (s)")
    ax_v.set_ylabel("Velocidad (mm/s)")
    ax_v.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig_v)

with col2:
    st.subheader("Perfil de Posición")
    fig_p, ax_p = plt.subplots(figsize=(6, 4))
    ax_p.plot(t, pos, color="tab:green", linewidth=2.5, label="Posición")
    ax_p.axhline(y=conveyor_length, color='red', linestyle='--', label='Fin de Conveyor (Stop)')
    ax_p.axhline(y=pos_sensor_red, color='orange', linestyle=':', label='Sensor Reducción')
    ax_p.set_xlabel("Tiempo (s)")
    ax_p.set_ylabel("Posición (mm)")
    ax_p.legend(loc="lower right")
    ax_p.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig_p)

# Métricas rápidas de resultado
st.markdown("---")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Tiempo Total de Ciclo", f"{t[-1]:.2f} s")
col_m2.metric("Distancia Sensor a Stop Configurada", f"{sensor_distance} mm")
col_m3.metric("Velocidad Máxima Alcanzada", f"{speed_fast} mm/s")
