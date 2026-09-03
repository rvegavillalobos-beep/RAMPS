import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Simulador Avanzado de Conveyor", layout="wide")

st.title("Simulador Dinámico de Perfiles de Aceleración y Frenado (Conveyor)")
st.markdown("Compara dos perfiles diferentes de forma fluida mediante animación nativa.")

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
    dt = 0.05  # Incrementado ligeramente para optimizar frames de animación
    t_max = 15.0
    steps = int(t_max / dt)
    
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)
    
    p = 0.0
    v = 0.0
    pos_sensor_red = length - s_dist
    pos_stop = length
    state = "ACCEL_FAST"
    
    for i in range(1, steps):
        t[i] = t[i-1] + dt
        
        if state == "ACCEL_FAST":
            v += accel * dt
            if v >= v_fast:
                v = v_fast
                state = "CRUISE_FAST"
        elif state == "CRUISE_FAST":
            if p >= pos_sensor_red:
                state = "DECEL_TO_SLOW"
        elif state == "DECEL_TO_SLOW":
            v -= decel * dt
            if v <= v_slow:
                v = v_slow
                state = "CRUISE_SLOW"
        elif state == "CRUISE_SLOW":
            dist_frenado_nec = (v_slow**2) / (2 * decel) if decel > 0 else 0
            if p >= (pos_stop - dist_frenado_nec):
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
            
    return t, pos, vel

t_a, pos_a, vel_a = calcular_perfil(v_fast_a, v_slow_a, acc_a, dec_a, conveyor_length, sensor_distance)
t_b, pos_b, vel_b = calcular_perfil(v_fast_b, v_slow_b, acc_b, dec_b, conveyor_length, sensor_distance)

# Normalizar longitudes de tiempo para la animación por frames
max_steps = max(len(t_a), len(t_b))
# Rellenar con el último valor estático para que ambas líneas duren el mismo número de pasos en la animación
t_anim = np.linspace(0, max(t_a[-1], t_b[-1]), max_steps)

pos_a_interp = np.interp(t_anim, t_a, pos_a)
vel_a_interp = np.interp(t_anim, t_a, vel_a)
pos_b_interp = np.interp(t_anim, t_b, pos_b)
vel_b_interp = np.interp(t_anim, t_b, vel_b)

# --- CONSTRUCCIÓN DE GRÁFICA CON FRAMES NATIVOS DE PLOTLY ---
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True,
    subplot_titles=("Perfil de Posición (Distancia vs Tiempo)", "Perfil de Velocidad (Velocidad vs Tiempo)"),
    vertical_spacing=0.15
)

# 1. Trazas de fondo estáticas (historial completo)
fig.add_trace(go.Scatter(x=t_a, y=pos_a, mode='lines', name='Historial A', line=dict(color='rgba(31, 119, 180, 0.3)', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=t_b, y=pos_b, mode='lines', name='Historial B', line=dict(color='rgba(255, 127, 14, 0.3)', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Hist. Vel A', line=dict(color='rgba(31, 119, 180, 0.3)', width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Hist. Vel B', line=dict(color='rgba(255, 127, 14, 0.3)', width=2)), row=2, col=1)

# 2. Trazas móviles iniciales (Frame 0)
fig.add_trace(go.Scatter(x=[pos_a_interp[0]], y=[vel_a_interp[0]], mode='markers', name='Pieza A', marker=dict(size=14, color='blue')), row=1, col=1) # Usaremos markers interactivos
# Para la posición en row=1:
fig.data[-1].x = [t_anim[0]]
fig.data[-1].y = [pos_a_interp[0]]

fig.add_trace(go.Scatter(x=[t_anim[0]], y=[pos_b_interp[0]], mode='markers', name='Pieza B', marker=dict(size=14, color='orange')), row=1, col=1)

fig.add_trace(go.Scatter(x=[t_anim[0]], y=[vel_a_interp[0]], mode='markers', name='Vel A', marker=dict(size=12, color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=[t_anim[0]], y=[vel_b_interp[0]], mode='markers', name='Vel B', marker=dict(size=12, color='orange')), row=2, col=1)

# Líneas de referencia
fig.add_hline(y=conveyor_length, line_dash="dash", line_color="red", annotation_text="Sensor Stop (Fin)", row=1, col=1)
fig.add_hline(y=conveyor_length - sensor_distance, line_dash="dot", line_color="orange", annotation_text="Sensor Reducción", row=1, col=1)

# Generar fotogramas para la animación nativa de Plotly
frames = []
for k in range(0, max_steps, 2):  # Salto de 2 para fluidez de rendimiento
    frames.append(go.Frame(
        data=[
            # Mantener trazas estáticas iguales
            go.Scatter(x=t_a, y=pos_a),
            go.Scatter(x=t_b, y=pos_b),
            go.Scatter(x=t_a, y=vel_a),
            go.Scatter(x=t_b, y=vel_b),
            # Actualizar marcadores móviles
            go.Scatter(x=[t_anim[k]], y=[pos_a_interp[k]]),
            go.Scatter(x=[t_anim[k]], y=[pos_b_interp[k]]),
            go.Scatter(x=[t_anim[k]], y=[vel_a_interp[k]]),
            go.Scatter(x=[t_anim[k]], y=[vel_b_interp[k]])
        ],
        name=str(k)
    ))

fig.frames = frames

# Configuración de Botones de Reproducción nativos dentro del gráfico
fig.update_layout(
    height=750,
    template="plotly_white",
    hovermode="x unified",
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "buttons": [
            {
                "label": "▶️ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 30, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]
            },
            {
                "label": "⏸️ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 10},
        "x": 0.1,
        "xanchor": "right",
        "y": 1.15,
        "yanchor": "top"
    }]
)

fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
fig.update_yaxes(title_text="Posición (mm)", row=1, col=1)
fig.update_yaxes(title_text="Velocidad (mm/s)", row=2, col=1)

# Renderizar en Streamlit sin parpadeos
st.plotly_chart(fig, use_container_width=True)
