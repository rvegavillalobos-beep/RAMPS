import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calculador de Perfil de Conveyor", layout="wide")

st.title("Calculador de Perfiles de Aceleración y Frenado - Conveyor")
st.markdown("Parámetros cinemáticos con marcas de eventos y tiempos de transición.")

# --- BARRA LATERAL DE PARÁMETROS ---
st.sidebar.header("Parámetros del Sistema")
speed_fast = st.sidebar.number_input("SPEED_AUTO_FAST (mm/s)", value=300.0, step=10.0)
speed_slow = st.sidebar.number_input("SPEED_AUTO_SLOW (mm/s)", value=100.0, step=10.0)
accel = st.sidebar.number_input("RAMP_ACCELERATION (mm/s²)", value=600.0, step=50.0)
decel = st.sidebar.number_input("RAMP_DECELERATION (mm/s²)", value=600.0, step=50.0)

st.sidebar.header("Geometría del Conveyor")
conveyor_length = st.sidebar.number_input("Largo total del conveyor (mm)", value=3000.0, step=100.0)
sensor_distance = st.sidebar.number_input("Distancia Sensor Reducción a Stop (mm)", value=800.0, step=50.0)

# --- CÁLCULO FÍSICO Y CAPTURA DE TIEMPOS DE EVENTO ---
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

t_accel_end = 0.0
t_sensor_red = 0.0
t_slow_reached = 0.0
t_brake_start = 0.0

for i in range(1, steps):
    t[i] = t[i-1] + dt
    
    if state == "ACCEL_FAST":
        a = accel
        v += a * dt
        if v >= speed_fast:
            v = speed_fast
            a = 0.0
            t_accel_end = t[i]
            state = "CRUISE_FAST"
            
    elif state == "CRUISE_FAST":
        a = 0.0
        if p >= pos_sensor_red:
            t_sensor_red = t[i]
            state = "DECEL_TO_SLOW"
            
    elif state == "DECEL_TO_SLOW":
        a = -decel
        v += a * dt
        if v <= speed_slow:
            v = speed_slow
            a = 0.0
            t_slow_reached = t[i]
            state = "CRUISE_SLOW"
            
    elif state == "CRUISE_SLOW":
        a = 0.0
        dist_frenado_nec = (speed_slow**2) / (2 * decel) if decel > 0 else 0
        if p >= (pos_stop - dist_frenado_nec):
            t_brake_start = t[i]
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

t_total = t[-1]
tiempo_aceleracion = t_accel_end
tiempo_desaceleracion_red = t_slow_reached - t_sensor_red
tiempo_slow_cruise = t_brake_start - t_slow_reached
tiempo_frenado_final = t_total - t_brake_start

# --- VISUALIZACIÓN EN GRÁFICAS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Perfil de Velocidad")
    fig_v, ax_v = plt.subplots(figsize=(6, 4.5))
    ax_v.plot(t, vel, color="tab:blue", linewidth=2.5, label="Velocidad")
    ax_v.axvline(x=t_sensor_red, color="orange", linestyle=":", label="Sensor Reducción")
    ax_v.axvline(x=t_brake_start, color="red", linestyle="--", label="Inicio Stop")
    
    ax_v.set_xlabel("Tiempo (s)")
    ax_v.set_ylabel("Velocidad (mm/s)")
    # Sacar la leyenda fuera de la gráfica (abajo, debajo del eje X)
    ax_v.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    ax_v.grid(True, linestyle="--", alpha=0.6)
    fig_v.subplots_adjust(bottom=0.2)  # Dar espacio para que no se corte
    st.pyplot(fig_v)

with col2:
    st.subheader("Perfil de Posición")
    fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
    ax_p.plot(t, pos, color="tab:green", linewidth=2.5, label="Posición")
    ax_p.axhline(y=conveyor_length, color='red', linestyle='--', label='Fin Conveyor (Stop)')
    ax_p.axhline(y=pos_sensor_red, color='orange', linestyle=':', label='Sensor Reducción')
    ax_p.axvline(x=t_sensor_red, color="orange", linestyle=":", alpha=0.7)
    ax_p.axvline(x=t_brake_start, color="red", linestyle="--", alpha=0.7)
    
    ax_p.set_xlabel("Tiempo (s)")
    ax_p.set_ylabel("Posición (mm)")
    # Sacar la leyenda fuera de la gráfica también
    ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    ax_p.grid(True, linestyle="--", alpha=0.6)
    fig_p.subplots_adjust(bottom=0.2)
    st.pyplot(fig_p)

# --- MÉTRICAS DETALLADAS DE TIEMPOS ---
st.markdown("---")
st.subheader("⏱️ Desglose de Tiempos del Ciclo")
col_m0, col_m1, col_m2, col_m3, col_m4 = st.columns(5)
col_m0.metric("Tiempo Total Ciclo", f"{t_total:.2f} s")
col_m1.metric("Aceleración", f"{tiempo_aceleracion:.2f} s")
col_m2.metric("Desacel. a Slow", f"{tiempo_desaceleracion_red:.2f} s")
col_m3.metric("Velocidad Slow", f"{tiempo_slow_cruise:.2f} s")
col_m4.metric("Frenado Final", f"{tiempo_frenado_final:.2f} s")
