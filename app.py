import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calculador de Perfil de Conveyor", layout="wide")

st.title("Calculador de Perfiles de Aceleración y Frenado - Conveyor")
st.markdown("Parámetros cinemáticos con identificación visual clara de sensores.")

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

# --- VISUALIZACIÓN EN GRÁFICAS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Perfil de Velocidad")
    fig_v, ax_v = plt.subplots(figsize=(6, 4.5))
    
    # Perfil A (Tonos Cálidos: Azul / Naranja / Rojo)
    ax_v.plot(t_a, vel_a, color="#1f77b4", linewidth=2.5, label="Velocidad A")
    ax_v.axvline(x=t_red_a, color="#ff7f0e", linestyle=":", linewidth=2, label="Reducción A")
    ax_v.axvline(x=t_stop_a, color="#d62728", linestyle="--", linewidth=2, label="Stop A")
    
    # Perfil B (Tonos Fríos y Contraste Alto: Morado / Cian / Magenta)
    if comparar:
        ax_v.plot(t_b, vel_b, color="#9467bd", linewidth=2.5, linestyle="-.", label="Velocidad B")
        ax_v.axvline(x=t_red_b, color="#17becf", linestyle=":", linewidth=2, label="Reducción B")
        ax_v.axvline(x=t_stop_b, color="#e377c2", linestyle="--", linewidth=2, label="Stop B")
        
    ax_v.set_xlabel("Tiempo (s)")
    ax_v.set_ylabel("Velocidad (mm/s)")
    ax_v.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    ax_v.grid(True, linestyle="--", alpha=0.6)
    fig_v.subplots_adjust(bottom=0.25)
    st.pyplot(fig_v)

with col2:
    st.subheader("Perfil de Posición")
    fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
    
    ax_p.axhline(y=conveyor_length, color='#d62728', linestyle='--', alpha=0.6, label='Fin Conveyor (Stop)')
    
    # Perfil A
    ax_p.plot(t_a, pos_a, color="#2ca02c", linewidth=2.5, label="Posición A")
    ax_p.axhline(y=conveyor_length - sensor_distance_a, color='#ff7f0e', linestyle=':', linewidth=2, label='Sensor Red. A')
    ax_p.axvline(x=t_red_a, color="#ff7f0e", linestyle=":", alpha=0.5)
    ax_p.axvline(x=t_stop_a, color="#d62728", linestyle="--", alpha=0.5)
    
    # Perfil B
    if comparar:
        ax_p.plot(t_b, pos_b, color="#8c564b", linewidth=2.5, linestyle="-.", label="Posición B")
        ax_p.axhline(y=conveyor_length - sensor_distance_b, color='#17becf', linestyle=':', linewidth=2, label='Sensor Red. B')
        ax_p.axvline(x=t_red_b, color="#17becf", linestyle=":", alpha=0.5)
        ax_p.axvline(x=t_stop_b, color="#e377c2", linestyle="--", alpha=0.5)
        
    ax_p.set_xlabel("Tiempo (s)")
    ax_p.set_ylabel("Posición (mm)")
    ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6.5)
    ax_p.grid(True, linestyle="--", alpha=0.6)
    fig_p.subplots_adjust(bottom=0.28)
    st.pyplot(fig_p)
