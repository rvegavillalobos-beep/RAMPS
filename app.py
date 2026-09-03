import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Rampas Conveyor", layout="wide")

st.title("Simulador de Perfiles de Aceleración y Frenado - Conveyor")
st.markdown("Ajusta los parámetros del sistema para simular el comportamiento cinemático.")

# --- BARRA LATERAL DE PARÁMETROS ---
st.sidebar.header("Parámetros del Sistema")

speed_fast = st.sidebar.number_input("SPEED_AUTO_FAST (mm/s)", value=300.0, step=10.0)
speed_slow = st.sidebar.number_input("SPEED_AUTO_SLOW (mm/s)", value=100.0, step=10.0)
accel = st.sidebar.number_input("RAMP_ACCELERATION (mm/s²)", value=600.0, step=50.0)
decel = st.sidebar.number_input("RAMP_DECELERATION (mm/s²)", value=600.0, step=50.0)

st.sidebar.header("Geometría del Conveyor")
conveyor_length = st.sidebar.number_input("Largo total del conveyor (mm)", value=3000.0, step=100.0)
sensor_distance = st.sidebar.number_input("Distancia Sensor Reducción a Stop (mm)", value=800.0, step=50.0)

# --- LÓGICA DE SIMULACIÓN ---
# Tiempo total de simulación y resolución temporal
dt = 0.01
t_max = 10.0
t = np.arange(0, t_max, dt)

# Simulación simplificada de un ciclo: Arranque -> Velocidad Fast -> Transición a Slow por Sensor -> Stop
# Definimos eventos basados en posición teórica
pos = np.zeros_like(t)
vel = np.zeros_like(t)
acc = np.zeros_like(t)

current_v = 0.0
current_p = 0.0
current_a = 0.0

# Máquina de estados simple para la simulación temporal
state = "ACCEL_TO_FAST"
for i in range(1, len(t)):
    # Lógica de estados de aceleración y desaceleración
    if state == "ACCEL_TO_FAST":
        current_a = accel
        current_v += current_a * dt
        if current_v >= speed_fast:
            current_v = speed_fast
            current_a = 0.0
            state = "CRUISE_FAST"
    elif state == "CRUISE_FAST":
        current_a = 0.0
        # Al llegar a la zona de desaceleración (ejemplo a 500mm del sensor de reducción)
        if current_p >= (conveyor_length - sensor_distance - 400):
            state = "DECEL_TO_SLOW"
    elif state == "DECEL_TO_SLOW":
        current_a = -decel
        current_v += current_a * dt
        if current_v <= speed_slow:
            current_v = speed_slow
            current_a = 0.0
            state = "CRUISE_SLOW"
    elif state == "CRUISE_SLOW":
        current_a = 0.0
        # Al llegar al sensor de stop final
        if current_p >= (conveyor_length - 200):
            state = "STOPPING"
    elif state == "STOPPING":
        current_a = -decel
        current_v += current_a * dt
        if current_v <= 0:
            current_v = 0.0
            current_a = 0.0
            state = "IDLE"
    elif state == "IDLE":
        current_a = 0.0
        current_v = 0.0

    current_p += current_v * dt
    vel[i] = current_v
    pos[i] = current_p
    acc[i] = current_a

# --- VISUALIZACIÓN DE GRÁFICAS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Perfil de Velocidad")
    fig_v, ax_v = plt.subplots(figsize=(6, 4))
    ax_v.plot(t, vel, color="blue", linewidth=2)
    ax_v.set_xlabel("Tiempo (s)")
    ax_v.set_ylabel("Velocidad (mm/s)")
    ax_v.grid(True)
    st.pyplot(fig_v)

with col2:
    st.subheader("Perfil de Posición")
    fig_p, ax_p = plt.subplots(figsize=(6, 4))
    ax_p.plot(t, pos, color="green", linewidth=2)
    ax_p.axhline(y=conveyor_length, color='r', linestyle='--', label='Fin de Conveyor')
    ax_p.set_xlabel("Tiempo (s)")
    ax_p.set_ylabel("Posición (mm)")
    ax_p.legend()
    ax_p.grid(True)
    st.pyplot(fig_p)
