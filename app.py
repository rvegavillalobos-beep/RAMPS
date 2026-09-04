import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calculador de Perfil de Conveyor", layout="wide")

st.title("Calculador de Perfiles de Aceleración y Frenado - Conveyor")
st.markdown("Parámetros cinemáticos con comparación opcional de doble perfil.")

# --- BARRA LATERAL DE PARÁMETROS ---
st.sidebar.header("Geometría Global")
conveyor_length = st.sidebar.number_input("Largo total del conveyor (mm)", value=3000.0, step=100.0)
sensor_distance = st.sidebar.number_input("Distancia Sensor Reducción a Stop (mm)", value=800.0, step=50.0)

st.sidebar.header("Perfil Principal (Perfil A)")
speed_fast_a = st.sidebar.number_input("SPEED_AUTO_FAST A (mm/s)", value=300.0, step=10.0)
speed_slow_a = st.sidebar.number_input("SPEED_AUTO_SLOW A (mm/s)", value=100.0, step=10.0)
accel_a = st.sidebar.number_input("RAMP_ACCEL A (mm/s²)", value=600.0, step=50.0)
decel_a = st.sidebar.number_input("RAMP_DECEL A (mm/s²)", value=600.0, step=50.0)

# Toggle para activar el segundo perfil
st.sidebar.markdown("---")
comparar = st.sidebar.checkbox("Comparar con Perfil B", value=False)

if comparar:
    st.sidebar.header("Perfil de Comparación (Perfil B)")
    speed_fast_b = st.sidebar.number_input("SPEED_AUTO_FAST B (mm/s)", value=450.0, step=10.0)
    speed_slow_b = st.sidebar.number_input("SPEED_AUTO_SLOW B (mm/s)", value=120.0, step=10.0)
    accel_b = st.sidebar.number_input("RAMP_ACCEL B (mm/s²)", value=800.0, step=50.0)
    decel_b = st.sidebar.number_input("RAMP_DECEL B (mm/s²)", value=400.0, step=50.0)

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

# Calcular Perfil A
t_a, pos_a, vel_a, t_red_a, t_stop_a = calcular_perfil(speed_fast_a, speed_slow_a, accel_a, decel_a, conveyor_length, sensor_distance)

# Calcular Perfil B si está activo
if comparar:
    t_b, pos_b, vel_b, t_red_b, t_stop_b = calcular_perfil(speed_fast_b, speed_slow_b, accel_b, decel_b, conveyor_length, sensor_distance)

# --- VISUALIZACIÓN EN GRÁFICAS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Perfil de Velocidad")
    fig_v, ax_v = plt.subplots(figsize=(6, 4.5))
    
    # Perfil A
    ax_v.plot(t_a, vel_a, color="tab:blue", linewidth=2.5, label="Velocidad A")
    ax_v.axvline(x=t_red_a, color="orange", linestyle=":", alpha=0.7, label="Sensor Reducción A")
    ax_v.axvline(x=t_stop_a, color="red", linestyle="--", alpha=0.7, label="Inicio Stop A")
    
    # Perfil B (si activo)
    if comparar:
        ax_v.plot(t_b, vel_b, color="tab:purple", linewidth=2.5, linestyle="-.", label="Velocidad B")
        ax_v.axvline(x=t_red_b, color="gold", linestyle=":", alpha=0.7, label="Sensor Reducción B")
        ax_v.axvline(x=t_stop_b, color="deeppink", linestyle="--", alpha=0.7, label="Inicio Stop B")
        
    ax_v.set_xlabel("Tiempo (s)")
    ax_v.set_ylabel("Velocidad (mm/s)")
    ax_v.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    ax_v.grid(True, linestyle="--", alpha=0.6)
    fig_v.subplots_adjust(bottom=0.25)
    st.pyplot(fig_v)

with col2:
    st.subheader("Perfil de Posición")
    fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
    
    # Referencias geométricas fijas
    ax_p.axhline(y=conveyor_length, color='red', linestyle='--', alpha=0.5, label='Fin Conveyor (Stop)')
    ax_p.axhline(y=conveyor_length - sensor_distance, color='orange', linestyle=':', alpha=0.5, label='Sensor Reducción')
    
    # Perfil A
    ax_p.plot(t_a, pos_a, color="tab:green", linewidth=2.5, label="Posición A")
    ax_p.axvline(x=t_red_a, color="orange", linestyle=":", alpha=0.5)
    ax_p.axvline(x=t_stop_a, color="red", linestyle="--", alpha=0.5)
    
    # Perfil B (si activo)
    if comparar:
        ax_p.plot(t_b, pos_b, color="tab:cyan", linewidth=2.5, linestyle="-.", label="Posición B")
        ax_p.axvline(x=t_red_b, color="gold", linestyle=":", alpha=0.5)
        ax_p.axvline(x=t_stop_b, color="deeppink", linestyle="--", alpha=0.5)
        
    ax_p.set_xlabel("Tiempo (s)")
    ax_p.set_ylabel("Posición (mm)")
    ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    ax_p.grid(True, linestyle="--", alpha=0.6)
    fig_p.subplots_adjust(bottom=0.25)
    st.pyplot(fig_p)

# --- MÉTRICAS COMPARATIVAS ---
st.markdown("---")
st.subheader("⏱️ Desglose de Tiempos del Ciclo")

if not comparar:
    col_m0, col_m1, col_m2, col_m3, col_m4 = st.columns(5)
    col_m0.metric("Tiempo Total Ciclo", f"{t_a[-1]:.2f} s")
    col_m1.metric("Aceleración", f"{t_red_a:.2f} s" if 't_red_a' in locals() else "0 s") # Simplificado de métrica pura
    col_m2.metric("Desacel. a Slow", f"{(t_stop_a - t_red_a):.2f} s")
    col_m3.metric("Velocidad Slow", f"{(t_a[-1] - t_stop_a):.2f} s")
    col_m4.metric("Frenado Final", f"{(t_a[-1] - t_stop_a):.2f} s")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Perfil A**")
        ca1, ca2, ca3 = st.columns(3)
        ca1.metric("Total Ciclo A", f"{t_a[-1]:.2f} s")
        ca2.metric("Máx Fast A", f"{speed_fast_a} mm/s")
        ca3.metric("Frenado A", f"{decel_a} mm/s²")
    with col_b:
        st.markdown("**Perfil B**")
        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Total Ciclo B", f"{t_b[-1]:.2f} s")
        cb2.metric("Máx Fast B", f"{speed_fast_b} mm/s")
        cb3.metric("Frenado B", f"{decel_b} mm/s²")
