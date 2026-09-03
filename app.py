import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Simulador Avanzado de Conveyor", layout="wide")

st.title("Simulador Dinámico de Perfiles de Aceleración y Frenado (Conveyor)")
st.markdown("Compara dos perfiles diferentes y visualiza la simulación cinemática en tiempo real.")

# --- BARRA LATERAL DE PARÁMETROS ---
st.sidebar.header("📏 Geometría Global")
conveyor_length = st.sidebar.number_input("Largo total del conveyor (mm)", value=3000.0, step=100.0)
sensor_distance = st.sidebar.number_input("Distancia Sensor Reducción a Stop (mm)", value=800.0, step=50.0)

col_cfg1, col_cfg2 = st.sidebar.columns(2)

with col_cfg1:
    st.subheader("Perfil A")
    v_fast_a = st.number_input("Fast A (mm/s)", value=300.0, step=10.0, key="v_fa")
    v_slow_a = st.number_input("Slow A (mm/s)", value=100.0, step=10.0, key="v_sa")
    acc_a = st.number_input("Accel A (mm/s²)", value=600.0, step=50.0, key="acc_a")
    dec_a = st.number_input("Decel A (mm/s²)", value=600.0, step=50.0, key="dec_a")

with col_cfg2:
    st.subheader("Perfil B (Comparación)")
    v_fast_b = st.number_input("Fast B (mm/s)", value=450.0, step=10.0, key="v_fb")
    v_slow_b = st.number_input("Slow B (mm/s)", value=120.0, step=10.0, key="v_sb")
    acc_b = st.number_input("Accel B (mm/s²)", value=800.0, step=50.0, key="acc_b")
    dec_b = st.number_input("Decel B (mm/s²)", value=400.0, step=50.0, key="dec_b")

# --- FUNCIÓN DE CÁLCULO CINEMÁTICO ---
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist):
    dt = 0.02
    t_max = 15.0
    steps = int(t_max / dt)
    
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)
    acc = np.zeros(steps)
    
    p = 0.0
    v = 0.0
    
    # Posiciones clave de los sensores
    # El sensor de reducción se encuentra a (length - s_dist)
    pos_sensor_red = length - s_dist
    pos_stop = length
    
    state = "ACCEL_FAST"
    
    for i in range(1, steps):
        t[i] = t[i-1] + dt
        
        # Máquina de estados realista del conveyor
        if state == "ACCEL_FAST":
            a = accel
            v += a * dt
            if v >= v_fast:
                v = v_fast
                a = 0.0
                state = "CRUISE_FAST"
                
        elif state == "CRUISE_FAST":
            a = 0.0
            # Al llegar al sensor de reducción de velocidad
            if p >= pos_sensor_red:
                state = "DECEL_TO_SLOW"
                
        elif state == "DECEL_TO_SLOW":
            a = -decel
            v += a * dt
            if v <= v_slow:
                v = v_slow
                a = 0.0
                state = "CRUISE_SLOW"
                
        elif state == "CRUISE_SLOW":
            a = 0.0
            # Calcular distancia exacta necesaria para frenar a 0 desde v_slow con la desaceleración dada
            dist_frenado_nec = (v_slow**2) / (2 * decel) if decel > 0 else 0
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
        # Evitar pasar del límite físico del conveyor en la simulación matemática
        if p > length:
            p = length
            v = 0.0
            a = 0.0
            
        pos[i] = p
        vel[i] = v
        acc[i] = a
        
        if state == "DONE" and i > 100 and np.all(vel[i-50:i] == 0):
            # Recortar arrays al tamaño real de la simulación para limpieza
            t = t[:i+1]
            pos = pos[:i+1]
            vel = vel[:i+1]
            acc = acc[:i+1]
            break
            
    return t, pos, vel, acc

t_a, pos_a, vel_a, acc_a_profile = calcular_perfil(v_fast_a, v_slow_a, acc_a, dec_a, conveyor_length, sensor_distance)
t_b, pos_b, vel_b, acc_b_profile = calcular_perfil(v_fast_b, v_slow_b, acc_b, dec_b, conveyor_length, sensor_distance)

# --- PANEL DE CONTROL DE ANIMACIÓN ---
st.markdown("---")
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 2, 1])
with col_ctrl1:
    btn_play = st.button("▶️ Reproducir Simulación en Vivo", type="primary")

# --- CONSTRUCCIÓN DE GRÁFICAS INTERACTIVAS (PLOTLY) ---
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True,
    subplot_titles=("Perfil de Posición (Distancia vs Tiempo)", "Perfil de Velocidad (Velocidad vs Tiempo)"),
    vertical_spacing=0.15
)

# Trazas estáticas completas (fondo tenue)
fig.add_trace(go.Scatter(x=t_a, y=pos_a, mode='lines', name='Posición A', line=dict(color='rgba(31, 119, 180, 0.4)', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=t_b, y=pos_b, mode='lines', name='Posición B', line=dict(color='rgba(255, 127, 14, 0.4)', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Velocidad A', line=dict(color='rgba(31, 119, 180, 0.4)', width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Velocidad B', line=dict(color='rgba(255, 127, 14, 0.4)', width=2)), row=2, col=1)

# Líneas de referencia de sensores en la gráfica de posición
fig.add_hline(y=conveyor_length, line_dash="dash", line_color="red", annotation_text="Sensor Stop (Fin)", row=1, col=1)
fig.add_hline(y=conveyor_length - sensor_distance, line_dash="dot", line_color="orange", annotation_text="Sensor Reducción", row=1, col=1)

# Trazas móviles (las "bolitas" que compiten)
fig.add_trace(go.Scatter(x=[t_a[0]], y=[pos_a[0]], mode='markers', name='Pieza A', marker=dict(size=12, color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=[t_b[0]], y=[pos_b[0]], mode='markers', name='Pieza B', marker=dict(size=12, color='orange')), row=1, col=1)

fig.add_trace(go.Scatter(x=[t_a[0]], y=[vel_a[0]], mode='markers', name='Vel A', marker=dict(size=10, color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=[t_b[0]], y=[vel_b[0]], mode='markers', name='Vel B', marker=dict(size=10, color='orange')), row=2, col=1)

fig.update_layout(height=700, template="plotly_white", hovermode="x unified")
fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
fig.update_yaxes(title_text="Posición (mm)", row=1, col=1)
fig.update_yaxes(title_text="Velocidad (mm/s)", row=2, col=1)

# Contenedor donde se dibuja el gráfico en Streamlit
chart_placeholder = st.empty()
chart_placeholder.plotly_chart(fig, use_container_width=True)

# --- LÓGICA DE ANIMACIÓN AL PRESIONAR PLAY ---
if btn_play:
    max_len = max(len(t_a), len(t_b))
    anim_placeholder = st.empty()
    
    # Bucle de animación en tiempo real simulado
    for i in range(0, max_len, 2): # Saltamos de 2 en 2 para fluidez
        # Obtener índices seguros para ambos perfiles
        idx_a = min(i, len(t_a) - 1)
        idx_b = min(i, len(t_b) - 1)
        
        # Copiamos la figura y actualizamos solo las posiciones de los marcadores móviles
        animated_fig = go.Figure(fig)
        
        # Actualizar marcadores de posición (índices 4 y 5)
        animated_fig.data[4].x = [t_a[idx_a]]
        animated_fig.data[4].y = [pos_a[idx_a]]
        animated_fig.data[5].x = [t_b[idx_b]]
        animated_fig.data[5].y = [pos_b[idx_b]]
        
        # Actualizar marcadores de velocidad (índices 6 y 7)
        animated_fig.data[6].x = [t_a[idx_a]]
        animated_fig.data[6].y = [vel_a[idx_a]]
        animated_fig.data[7].x = [t_b[idx_b]]
        animated_fig.data[7].y = [vel_b[idx_b]]
        
        chart_placeholder.plotly_chart(animated_fig, use_container_width=True, key=f"anim_{i}")
        time.sleep(0.01) # Control de velocidad de reproducción en pantalla
