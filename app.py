import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ==========================================
# BACKSLASH-SAFE LATEX HELPER
# ==========================================
BS = chr(92)

def L(template: str) -> str:
    """Reemplaza '~' por un backslash real para construir LaTeX válido."""
    return template.replace("~", BS)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Kinematic & Physics Conveyor Simulator", layout="wide")
st.title("Kinematic & Physics Conveyor Simulator")
st.markdown("Real time analysis of velocity profile, positioning, and inertial part slip.")

# ==========================================
# CONSTANTES FÍSICAS DEL CARRIER Y LA CARGA (NO CONFIGURABLES)
# ==========================================
CARRIER_ALTO_MM = 250.0
CARRIER_LARGO_MM = 1600.0
CARGA_ALTO_MM = 130.0
CARGA_LARGO_MM = 1200.0
PIEZA_MESA_ANCHO_MM = 400.0

# ==========================================
# CONVERSIÓN ÁNGULO <-> MM SOBRE UN RIEL A RADIO FIJO
# ==========================================
def angle_to_mm(angle_deg, radius, mode="Arco (R·θ)"):
    theta_rad = np.radians(angle_deg)
    if mode.startswith("Arco"):
        return radius * theta_rad
    else:
        return 2.0 * radius * np.sin(theta_rad / 2.0)

def mm_to_angle(mm, radius, mode="Arco (R·θ)"):
    if radius <= 0:
        return 0.0
    if mode.startswith("Arco"):
        theta_rad = mm / radius
    else:
        ratio = np.clip(mm / (2.0 * radius), -1.0, 1.0)
        theta_rad = 2.0 * np.arcsin(ratio)
    return np.degrees(theta_rad)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
defaults = {
    "conveyor_length": 3000.0,
    "speed_fast_a": 300.0, "speed_slow_a": 100.0, "accel_a": 300.0, "decel_a": 300.0,
    "sensor_distance_a": 150.0, "ramp_stop_a": 150.0, "mu_a": 0.28,
    "comparar": False,
    "speed_fast_b": 450.0, "speed_slow_b": 120.0, "accel_b": 400.0, "decel_b": 300.0,
    "sensor_distance_b": 150.0, "ramp_stop_b": 0.0, "mu_b": 0.28,
    "speed_fast_mesa": 45.0, "speed_slow_mesa": 10.0, "accel_mesa": 90.0, "decel_mesa": 90.0,
    "sensor_angle_mesa": 15.0, "ramp_stop_mesa": 100.0, "mu_mesa": 0.28,
    "angle_total_mesa": 90.0, "pieza_longitud_mesa": 2110.0,
    "radio_sensor_mesa": 545.25,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
if "_comparar_prev" not in st.session_state:
    st.session_state._comparar_prev = False

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📐 Geometría Global")
st.sidebar.number_input("Largo Total Conveyor (mm)", value=st.session_state.conveyor_length, step=100.0, key="conveyor_length")

st.sidebar.header("🔵 Perfil A (Principal)")
st.sidebar.number_input("SPEED_AUTO_FAST A (mm/s)", value=st.session_state.speed_fast_a, step=10.0, key="speed_fast_a")
st.sidebar.number_input("SPEED_AUTO_SLOW A (mm/s)", value=st.session_state.speed_slow_a, step=10.0, key="speed_slow_a")
st.sidebar.number_input("RAMP_ACCEL A (mm/s²)", value=st.session_state.accel_a, step=50.0, key="accel_a")
st.sidebar.number_input("RAMP_DECEL A (mm/s²)", value=st.session_state.decel_a, step=50.0, key="decel_a")
st.sidebar.number_input("Distancia Sensor Reducción A (mm)", value=st.session_state.sensor_distance_a, step=25.0, key="sensor_distance_a")
st.sidebar.number_input("RAMP_STOP A (ms)", value=st.session_state.ramp_stop_a, step=10.0, min_value=0.0, key="ramp_stop_a", help="Rampa de frenado en el sensor de paro")
st.sidebar.number_input("Coeficiente Fricción μ A", value=st.session_state.mu_a, step=0.01, min_value=0.01, max_value=1.0, key="mu_a")

st.sidebar.markdown("---")
st.sidebar.checkbox("Comparar con Perfil B", value=st.session_state.comparar, key="comparar")

if st.session_state.comparar and not st.session_state._comparar_prev:
    st.session_state.speed_fast_b = st.session_state.speed_fast_a
    st.session_state.speed_slow_b = st.session_state.speed_slow_a
    st.session_state.accel_b = st.session_state.accel_a
    st.session_state.decel_b = st.session_state.decel_a
    st.session_state.sensor_distance_b = st.session_state.sensor_distance_a
    st.session_state.ramp_stop_b = st.session_state.ramp_stop_a
    st.session_state.mu_b = st.session_state.mu_a
st.session_state._comparar_prev = st.session_state.comparar

if st.session_state.comparar:
    st.sidebar.header("🟣 Perfil B (Comparativa)")
    st.sidebar.number_input("SPEED_AUTO_FAST B (mm/s)", value=st.session_state.speed_fast_b, step=10.0, key="speed_fast_b")
    st.sidebar.number_input("SPEED_AUTO_SLOW B (mm/s)", value=st.session_state.speed_slow_b, step=10.0, key="speed_slow_b")
    st.sidebar.number_input("RAMP_ACCEL B (mm/s²)", value=st.session_state.accel_b, step=50.0, key="accel_b")
    st.sidebar.number_input("RAMP_DECEL B (mm/s²)", value=st.session_state.decel_b, step=50.0, key="decel_b")
    st.sidebar.number_input("Distancia Sensor Reducción B (mm)", value=st.session_state.sensor_distance_b, step=25.0, key="sensor_distance_b")
    st.sidebar.number_input("RAMP_STOP B (ms)", value=st.session_state.ramp_stop_b, step=10.0, min_value=0.0, key="ramp_stop_b")
    st.sidebar.number_input("Coeficiente Fricción μ B", value=st.session_state.mu_b, step=0.01, min_value=0.01, max_value=1.0, key="mu_b")

st.sidebar.markdown("---")
st.sidebar.header("🔄 Mesa Giratoria (Turning Table)")
st.sidebar.number_input("SPEED_AUTO_FAST MESA (°/s)", value=st.session_state.speed_fast_mesa, step=5.0, key="speed_fast_mesa")
st.sidebar.number_input("SPEED_AUTO_SLOW MESA (°/s)", value=st.session_state.speed_slow_mesa, step=1.0, key="speed_slow_mesa")
st.sidebar.number_input("RAMP_ACCEL MESA (°/s²)", value=st.session_state.accel_mesa, step=10.0, key="accel_mesa")
st.sidebar.number_input("RAMP_DECEL MESA (°/s²)", value=st.session_state.decel_mesa, step=10.0, key="decel_mesa")
st.sidebar.number_input("Ángulo Sensor Reducción MESA (°)", value=st.session_state.sensor_angle_mesa, step=5.0, key="sensor_angle_mesa")
st.sidebar.number_input("RAMP_STOP MESA (ms)", value=st.session_state.ramp_stop_mesa, step=10.0, min_value=0.0, key="ramp_stop_mesa", help="Rampa de frenado angular en el sensor de paro")
st.sidebar.number_input("Coeficiente Fricción μ MESA", value=st.session_state.mu_mesa, step=0.01, min_value=0.01, max_value=1.0, key="mu_mesa")
st.sidebar.number_input("Ángulo Total de Giro MESA (°)", value=st.session_state.angle_total_mesa, step=15.0, key="angle_total_mesa", help="Equivalente angular de 'Largo Total Conveyor' — abierto")
st.sidebar.number_input("Longitud Total de Pieza MESA (mm)", value=st.session_state.pieza_longitud_mesa, step=10.0, min_value=1.0, key="pieza_longitud_mesa", help="La pieza se posiciona centrada en el eje de giro. R_max = Longitud / 2 (usado SOLO para el análisis de deslizamiento)")
st.sidebar.number_input("Radio Riel Sensores MESA (mm)", value=st.session_state.radio_sensor_mesa, step=5.0, min_value=1.0, key="radio_sensor_mesa", help="Radio físico donde están montados los sensores de reducción/paro sobre el riel — distinto del radio de la pieza")


# ==========================================
# REALISTIC KINEMATIC CALCULATION ENGINE (CONVEYOR LINEAL)
# ==========================================
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist, ramp_stop_ms, mu):
    dt = 0.001
    t_max = 30.0
    steps = int(t_max / dt)
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)

    p = 0.0
    v = 0.0
    pos_sensor_red = length - s_dist
    pos_stop = length
    state = "ACCEL_FAST"

    t_reach_fast = 0.0
    t_sensor_red = 0.0
    t_reach_slow = 0.0
    t_sensor_stop = 0.0
    t_fully_stopped = 0.0

    T_MIN_MECANICO_MS = 20.0
    ramp_stop_real_ms = max(ramp_stop_ms, T_MIN_MECANICO_MS)

    a_stop_conveyor = v_slow / (ramp_stop_real_ms / 1000.0)
    g_conveyor = a_stop_conveyor / 9810.0
    g_max_pieza = mu
    a_max_pieza = mu * 9810.0

    if v_fast > v_slow and decel > 0:
        dist_needed_decel = (v_fast ** 2 - v_slow ** 2) / (2.0 * decel)
    else:
        dist_needed_decel = 0.0
    insufficient_distance = dist_needed_decel > s_dist

    for i in range(1, steps):
        t[i] = t[i - 1] + dt

        if state not in ("DECEL_TO_STOP", "DONE") and p >= pos_stop:
            t_sensor_stop = t[i]
            state = "DECEL_TO_STOP"

        if state == "ACCEL_FAST":
            v += accel * dt
            if v >= v_fast:
                v = v_fast
                t_reach_fast = t[i]
                state = "CRUISE_FAST"
        elif state == "CRUISE_FAST":
            if p >= pos_sensor_red:
                t_sensor_red = t[i]
                state = "DECEL_TO_SLOW"
        elif state == "DECEL_TO_SLOW":
            v -= decel * dt
            if v <= v_slow:
                v = v_slow
                t_reach_slow = t[i]
                state = "CRUISE_SLOW"
        elif state == "CRUISE_SLOW":
            pass
        elif state == "DECEL_TO_STOP":
            v -= a_stop_conveyor * dt
            if v <= 0:
                v = 0.0
                t_fully_stopped = t[i]
                state = "DONE"
        elif state == "DONE":
            v = 0.0

        p += v * dt
        pos[i] = p
        vel[i] = v

        if state == "DONE" and i > 50 and np.all(vel[i - 20:i] == 0):
            t = t[:i + 1]
            pos = pos[:i + 1]
            vel = vel[:i + 1]
            break

    if t_reach_slow == 0.0 and t_sensor_stop > 0.0:
        t_reach_slow = t_sensor_stop

    dist_overrun_conveyor = pos[-1] - pos_stop
    overrun_time = max(t_fully_stopped - t_sensor_stop, 0.0)

    se_desliza = g_conveyor > g_max_pieza
    if se_desliza:
        dist_freno_pieza = (v_slow ** 2) / (2.0 * a_max_pieza)
        dist_freno_conveyor = (v_slow ** 2) / (2.0 * a_stop_conveyor)
        deslizamiento_mm = dist_freno_pieza - dist_freno_conveyor
        g_pieza_real = g_max_pieza
    else:
        dist_freno_pieza = (v_slow ** 2) / (2.0 * a_stop_conveyor)
        dist_freno_conveyor = dist_freno_pieza
        deslizamiento_mm = 0.0
        g_pieza_real = g_conveyor

    factor_seguridad = (mu / g_conveyor) if g_conveyor > 0 else float('inf')
    mu_minimo_requerido = g_conveyor

    details = {
        "ramp_stop_real_ms": ramp_stop_real_ms,
        "a_stop_conveyor": a_stop_conveyor,
        "a_max_pieza": a_max_pieza,
        "dist_freno_pieza": dist_freno_pieza,
        "dist_freno_conveyor": dist_freno_conveyor,
        "t_reach_fast": t_reach_fast,
        "t_reach_slow": t_reach_slow,
        "t_fully_stopped": t_fully_stopped,
        "factor_seguridad": factor_seguridad,
        "mu_minimo_requerido": mu_minimo_requerido,
        "pos_sensor_red": pos_sensor_red,
        "pos_stop": pos_stop,
        "insufficient_distance": insufficient_distance,
        "dist_needed_decel": dist_needed_decel,
    }

    return (t, pos, vel, t_sensor_red, t_sensor_stop, dist_overrun_conveyor, overrun_time,
            g_conveyor, g_pieza_real, se_desliza, deslizamiento_mm, details)


(t_a, pos_a, vel_a, t_red_a, t_stop_a, overrun_a, overrun_time_a,
 g_conv_a, g_pieza_a, desliza_a, d_desliza_a, det_a) = calcular_perfil(
    st.session_state.speed_fast_a, st.session_state.speed_slow_a, st.session_state.accel_a,
    st.session_state.decel_a, st.session_state.conveyor_length, st.session_state.sensor_distance_a,
    st.session_state.ramp_stop_a, st.session_state.mu_a
)

if st.session_state.comparar:
    (t_b, pos_b, vel_b, t_red_b, t_stop_b, overrun_b, overrun_time_b,
     g_conv_b, g_pieza_b, desliza_b, d_desliza_b, det_b) = calcular_perfil(
        st.session_state.speed_fast_b, st.session_state.speed_slow_b, st.session_state.accel_b,
        st.session_state.decel_b, st.session_state.conveyor_length, st.session_state.sensor_distance_b,
        st.session_state.ramp_stop_b, st.session_state.mu_b
    )


# ==========================================
# ANGULAR KINEMATIC CALCULATION ENGINE (MESA GIRATORIA)
# ==========================================
def calcular_perfil_mesa(omega_fast, omega_slow, alpha_accel, alpha_decel,
                          angle_total, angle_sensor_dist, ramp_stop_ms, mu, r_max):
    dt = 0.001
    t_max = 30.0
    steps = int(t_max / dt)
    t = np.zeros(steps)
    theta = np.zeros(steps)
    omega = np.zeros(steps)

    th = 0.0
    om = 0.0
    angle_sensor_red = angle_total - angle_sensor_dist
    angle_stop = angle_total
    state = "ACCEL_FAST"

    t_reach_fast = 0.0
    t_sensor_red = 0.0
    t_reach_slow = 0.0
    t_sensor_stop = 0.0
    t_fully_stopped = 0.0

    T_MIN_MECANICO_MS = 20.0
    ramp_stop_real_ms = max(ramp_stop_ms, T_MIN_MECANICO_MS)

    alpha_stop = omega_slow / (ramp_stop_real_ms / 1000.0)
    alpha_stop_rad = np.radians(alpha_stop)
    a_tan_stop = alpha_stop_rad * r_max
    g_conv_mesa = a_tan_stop / 9810.0

    g_max_pieza = mu
    a_max_pieza = mu * 9810.0

    omega_fast_rad = np.radians(omega_fast)
    omega_slow_rad = np.radians(omega_slow)
    a_cent_fast = (omega_fast_rad ** 2) * r_max
    a_cent_slow = (omega_slow_rad ** 2) * r_max
    g_cent_fast = a_cent_fast / 9810.0
    g_cent_slow = a_cent_slow / 9810.0

    if omega_fast > omega_slow and alpha_decel > 0:
        angle_needed_decel = (omega_fast ** 2 - omega_slow ** 2) / (2.0 * alpha_decel)
    else:
        angle_needed_decel = 0.0
    insufficient_angle = angle_needed_decel > angle_sensor_dist

    for i in range(1, steps):
        t[i] = t[i - 1] + dt

        if state not in ("DECEL_TO_STOP", "DONE") and th >= angle_stop:
            t_sensor_stop = t[i]
            state = "DECEL_TO_STOP"

        if state == "ACCEL_FAST":
            om += alpha_accel * dt
            if om >= omega_fast:
                om = omega_fast
                t_reach_fast = t[i]
                state = "CRUISE_FAST"
        elif state == "CRUISE_FAST":
            if th >= angle_sensor_red:
                t_sensor_red = t[i]
                state = "DECEL_TO_SLOW"
        elif state == "DECEL_TO_SLOW":
            om -= alpha_decel * dt
            if om <= omega_slow:
                om = omega_slow
                t_reach_slow = t[i]
                state = "CRUISE_SLOW"
        elif state == "CRUISE_SLOW":
            pass
        elif state == "DECEL_TO_STOP":
            om -= alpha_stop * dt
            if om <= 0:
                om = 0.0
                t_fully_stopped = t[i]
                state = "DONE"
        elif state == "DONE":
            om = 0.0

        th += om * dt
        theta[i] = th
        omega[i] = om

        if state == "DONE" and i > 50 and np.all(omega[i - 20:i] == 0):
            t = t[:i + 1]
            theta = theta[:i + 1]
            omega = omega[:i + 1]
            break

    if t_reach_slow == 0.0 and t_sensor_stop > 0.0:
        t_reach_slow = t_sensor_stop

    angle_overrun = theta[-1] - angle_stop
    overrun_time = max(t_fully_stopped - t_sensor_stop, 0.0)

    v_slow_lineal = omega_slow_rad * r_max
    se_desliza_stop = g_conv_mesa > g_max_pieza
    if se_desliza_stop:
        dist_freno_pieza = (v_slow_lineal ** 2) / (2.0 * a_max_pieza)
        dist_freno_mesa = (v_slow_lineal ** 2) / (2.0 * a_tan_stop) if a_tan_stop > 0 else 0.0
        deslizamiento_mm_stop = dist_freno_pieza - dist_freno_mesa
    else:
        dist_freno_pieza = (v_slow_lineal ** 2) / (2.0 * a_tan_stop) if a_tan_stop > 0 else 0.0
        dist_freno_mesa = dist_freno_pieza
        deslizamiento_mm_stop = 0.0
    deslizamiento_deg_stop = np.degrees(deslizamiento_mm_stop / r_max) if r_max > 0 else 0.0

    alpha_max_pieza_rad = (a_max_pieza / r_max) if r_max > 0 else 0.0
    alpha_max_pieza_deg = np.degrees(alpha_max_pieza_rad)

    se_desliza_cruise_fast = g_cent_fast > mu
    se_desliza_cruise_slow = g_cent_slow > mu

    factor_seguridad_stop = (mu / g_conv_mesa) if g_conv_mesa > 0 else float('inf')
    factor_seguridad_cruise_fast = (mu / g_cent_fast) if g_cent_fast > 0 else float('inf')

    details = {
        "ramp_stop_real_ms": ramp_stop_real_ms,
        "alpha_stop": alpha_stop,
        "a_tan_stop": a_tan_stop,
        "a_max_pieza": a_max_pieza,
        "dist_freno_pieza": dist_freno_pieza,
        "dist_freno_mesa": dist_freno_mesa,
        "deslizamiento_deg_stop": deslizamiento_deg_stop,
        "alpha_max_pieza_deg": alpha_max_pieza_deg,
        "t_reach_fast": t_reach_fast,
        "t_reach_slow": t_reach_slow,
        "t_fully_stopped": t_fully_stopped,
        "angle_sensor_red": angle_sensor_red,
        "angle_stop": angle_stop,
        "insufficient_angle": insufficient_angle,
        "angle_needed_decel": angle_needed_decel,
        "g_cent_fast": g_cent_fast,
        "g_cent_slow": g_cent_slow,
        "se_desliza_cruise_fast": se_desliza_cruise_fast,
        "se_desliza_cruise_slow": se_desliza_cruise_slow,
        "factor_seguridad_stop": factor_seguridad_stop,
        "factor_seguridad_cruise_fast": factor_seguridad_cruise_fast,
        "v_slow_lineal": v_slow_lineal,
    }

    return (t, theta, omega, t_sensor_red, t_sensor_stop, angle_overrun, overrun_time,
            g_conv_mesa, se_desliza_stop, deslizamiento_mm_stop, details)


r_max_mesa = st.session_state.pieza_longitud_mesa / 2.0

(t_m, theta_m, omega_m, t_red_m, t_stop_m, ang_overrun_m, overrun_time_m,
 g_conv_m, desliza_m, d_desliza_mm_m, det_m) = calcular_perfil_mesa(
    st.session_state.speed_fast_mesa, st.session_state.speed_slow_mesa,
    st.session_state.accel_mesa, st.session_state.decel_mesa,
    st.session_state.angle_total_mesa, st.session_state.sensor_angle_mesa,
    st.session_state.ramp_stop_mesa, st.session_state.mu_mesa, r_max_mesa
)


tab_sim, tab_math = st.tabs(["📊 Simulation & Dashboard", "📚 Mathematical Background & Physics Engine"])

# ==========================================
# TAB 1: SIMULATION & DASHBOARD
# ==========================================
with tab_sim:

    if det_a['insufficient_distance']:
        st.warning(
            f"⚠️ **Perfil A — Configuración inconsistente:** la rampa `RAMP_DECEL A` necesita "
            f"**{det_a['dist_needed_decel']:.1f} mm** para bajar de SPEED_AUTO_FAST a SPEED_AUTO_SLOW, "
            f"pero la `Distancia Sensor Reducción A` solo tiene **{st.session_state.sensor_distance_a:.1f} mm**."
        )
    if st.session_state.comparar and det_b['insufficient_distance']:
        st.warning(
            f"⚠️ **Perfil B — Configuración inconsistente:** la rampa `RAMP_DECEL B` necesita "
            f"**{det_b['dist_needed_decel']:.1f} mm** para bajar de SPEED_AUTO_FAST a SPEED_AUTO_SLOW, "
            f"pero la `Distancia Sensor Reducción B` solo tiene **{st.session_state.sensor_distance_b:.1f} mm**."
        )

    # ---- 1. GRÁFICA DE VELOCIDAD (LINEAL) ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Velocity Profile A',
                              line=dict(color='#1f77b4', width=3)))
    if st.session_state.comparar:
        fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Velocity Profile B',
                                  line=dict(color='#9467bd', width=3, dash='dashdot')))

    fig.add_vline(x=t_red_a, line_width=2, line_dash="dot", line_color="#ff7f0e",
                  annotation_text=f"Sensor Reducción A · {t_red_a:.2f}s",
                  annotation_position="top left", annotation_font_size=10, annotation_font_color="#ff7f0e")
    fig.add_vline(x=t_stop_a, line_width=2, line_dash="dash", line_color="#d62728",
                  annotation_text=f"Sensor Paro A · {t_stop_a:.2f}s",
                  annotation_position="top right", annotation_font_size=10, annotation_font_color="#d62728")
    if st.session_state.comparar:
        fig.add_vline(x=t_red_b, line_width=2, line_dash="dot", line_color="#9467bd",
                      annotation_text=f"Sensor Reducción B · {t_red_b:.2f}s",
                      annotation_position="bottom left", annotation_font_size=10, annotation_font_color="#9467bd")
        fig.add_vline(x=t_stop_b, line_width=2, line_dash="dash", line_color="#6f4a94",
                      annotation_text=f"Sensor Paro B · {t_stop_b:.2f}s",
                      annotation_position="bottom right", annotation_font_size=10, annotation_font_color="#6f4a94")

    fig.update_layout(
        title="Velocity Profile (mm/s) vs Time (s)",
        xaxis_title="Time (s)", yaxis_title="Velocity (mm/s)",
        height=480, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(b=110, t=90)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 2. SLIP ANALYSIS ----
    st.markdown("---")
    st.subheader("📊 Inertia & Part Slip Analysis (Profile A)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conveyor Deceleration", f"{g_conv_a:.3f} G")
    col2.metric("Friction Limit (μ)", f"{st.session_state.mu_a:.2f} G")
    col3.metric("Load Stability", "🔴 SLIPPING" if desliza_a else "🟢 STABLE")
    col4.metric("Relative Part Slip", f"{d_desliza_a:.3f} mm ({d_desliza_a*1000:.0f} µm)" if desliza_a else "0.000 mm")

    col_s1, col_s2 = st.columns(2)
    fs = det_a['factor_seguridad']
    fs_display = "∞" if fs == float('inf') else f"{fs:.2f}x"
    col_s1.metric("🛡️ Factor de Seguridad (μ / g_conv)", fs_display)
    col_s2.metric("🎯 μ mínimo requerido para NO deslizar", f"{det_a['mu_minimo_requerido']:.3f}")

    expander_title = "🔍 View Calculation Step by Step Breakdown (Profile A)" if desliza_a else "ℹ️ View Stability & Deceleration Math (Profile A)"
    with st.expander(expander_title):
        st.markdown("### 🧮 Live Calculation Breakdown (Simulated Values)")

        st.markdown("**Step 1: Effective Stop Ramp Time**")
        st.latex(L(
            f"t_{{~text{{stop~_real}}}} = ~max({st.session_state.ramp_stop_a:.1f}~,~text{{ms}}, "
            f"20.0~,~text{{ms}}) = {det_a['ramp_stop_real_ms']:.1f}~,~text{{ms}} = "
            f"{det_a['ramp_stop_real_ms']/1000.0:.3f}~,~text{{s}}"
        ))

        st.markdown("**Step 2: Conveyor Stop Deceleration**")
        st.latex(L(
            f"a_{{~text{{stop}}}} = ~frac{{{st.session_state.speed_slow_a:.1f}~,~text{{mm/s}}}}"
            f"{{{det_a['ramp_stop_real_ms']/1000.0:.3f}~,~text{{s}}}} = {det_a['a_stop_conveyor']:.2f}~,~text{{mm/s}}^2"
        ))
        st.latex(L(
            f"g_{{~text{{conv}}}} = ~frac{{{det_a['a_stop_conveyor']:.2f}~,~text{{mm/s}}^2}}"
            f"{{9810~,~text{{mm/s}}^2}} = ~mathbf{{{g_conv_a:.3f}~,~text{{G}}}}"
        ))

        st.markdown("**Step 3: Maximum Allowable Friction Acceleration**")
        st.latex(L(
            f"a_{{~text{{max~_piece}}}} = ~mu ~cdot g = {st.session_state.mu_a:.2f} ~cdot 9810~,~text{{mm/s}}^2 "
            f"= {det_a['a_max_pieza']:.2f}~,~text{{mm/s}}^2 ~quad (~mu = ~mathbf{{{st.session_state.mu_a:.2f}~,~text{{G}}}})"
        ))

        st.markdown("**Step 4: Slip Decision Criteria**")
        if desliza_a:
            st.error(
                f"🔴 **SLIP DETECTED:** g_conv ({g_conv_a:.3f} G) > μ ({st.session_state.mu_a:.2f} G)."
            )
            st.markdown("**Step 5: Relative Slip Distance Calculation (Δd)**")
            st.latex(L(
                f"d_{{~text{{piece}}}} = ~frac{{v_{{~text{{slow}}}}^2}}{{2 ~cdot a_{{~text{{max~_piece}}}}}} = "
                f"{det_a['dist_freno_pieza']:.3f}~,~text{{mm}}"
            ))
            st.latex(L(
                f"d_{{~text{{conveyor}}}} = ~frac{{v_{{~text{{slow}}}}^2}}{{2 ~cdot a_{{~text{{stop}}}}}} = "
                f"{det_a['dist_freno_conveyor']:.3f}~,~text{{mm}}"
            ))
            st.latex(L(
                f"~Delta d = {det_a['dist_freno_pieza']:.3f} - {det_a['dist_freno_conveyor']:.3f} = "
                f"~mathbf{{{d_desliza_a:.3f}~,~text{{mm}}}}"
            ))
        else:
            st.success(
                f"🟢 **STABLE LOAD:** g_conv ({g_conv_a:.3f} G) ≤ μ ({st.session_state.mu_a:.2f} G)."
            )

    # ---- 3. Positioning & Cycle Time ----
    st.markdown("---")
    st.subheader("🎯 Positioning & Cycle Time (Profile A)")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Conveyor Overrun", f"{overrun_a:.2f} mm")
    col6.metric("Overrun Time", f"{overrun_time_a*1000:.0f} ms")
    col7.metric("Final Part Position", f"{(pos_a[-1] + d_desliza_a):.2f} mm")
    col8.metric("Total Motion Time", f"{t_a[-1]:.2f} s")

    if st.session_state.comparar:
        st.markdown("---")
        st.subheader("⚖️ Profile A vs Profile B Comparison")
        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        c_b1.metric("Deceleration Profile B", f"{g_conv_b:.3f} G", delta=f"{(g_conv_b - g_conv_a):.3f} G", delta_color="inverse")
        c_b2.metric("Part Slip Profile B", f"{d_desliza_b:.3f} mm", delta=f"{(d_desliza_b - d_desliza_a):.3f} mm", delta_color="inverse")
        c_b3.metric("Overrun Profile B", f"{overrun_b:.2f} mm", delta=f"{(overrun_b - overrun_a):.2f} mm", delta_color="inverse")
        c_b4.metric("Overrun Time Profile B", f"{overrun_time_b*1000:.0f} ms", delta=f"{(overrun_time_b - overrun_time_a)*1000:.0f} ms", delta_color="inverse")

        c_b5, c_b6 = st.columns(2)
        c_b5.metric("Final Part Pos B", f"{(pos_b[-1] + d_desliza_b):.2f} mm",
                    delta=f"{(pos_b[-1] + d_desliza_b) - (pos_a[-1] + d_desliza_a):.2f} mm", delta_color="inverse")
        c_b6.metric("Cycle Time Profile B", f"{t_b[-1]:.2f} s", delta=f"{(t_b[-1] - t_a[-1]):.2f} s", delta_color="inverse")

    # ==========================================
    # 5. ANIMACIÓN — Riel General (Calidad Alta fija)
    # ==========================================
    st.markdown("---")
    st.subheader("🎬 Animación en Tiempo Real: Recorrido de la Pieza")
    st.caption("El riel se colorea conforme la pieza avanza. Calidad fijada en Alta (260 frames).")

    velocidad_reproduccion = st.select_slider("Velocidad de reproducción", options=["0.5x", "1x", "2x", "4x"], value="1x", key="vel_riel")
    speed_map = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
    speed_mult = speed_map[velocidad_reproduccion]
    n_frames = 260

    STATE_INFO = {
        "ACCEL_FAST":    ("🚀", "Acelerando"),
        "CRUISE_FAST":   ("➡️", "Crucero Rápido"),
        "DECEL_TO_SLOW": ("🐢", "Frenando a Lento"),
        "CRUISE_SLOW":   ("🐌", "Crucero Lento"),
        "DECEL_TO_STOP": ("🛑", "Frenado Final"),
        "DONE":          ("✅", "Detenido"),
    }
    STATE_COLOR = {
        "ACCEL_FAST": "#2ecc71", "CRUISE_FAST": None,
        "DECEL_TO_SLOW": "#f39c12", "CRUISE_SLOW": "#f1c40f",
        "DECEL_TO_STOP": "#e74c3c", "DONE": "#7f8c8d",
    }

    def get_state(ft, t_rf, t_red, t_rs, t_stop, t_full):
        if ft < t_rf: return "ACCEL_FAST"
        if ft < t_red: return "CRUISE_FAST"
        if ft < t_rs: return "DECEL_TO_SLOW"
        if ft < t_stop: return "CRUISE_SLOW"
        if ft < t_full: return "DECEL_TO_STOP"
        return "DONE"

    def construir_track(perfiles, length, n_frames, speed_mult):
        t_end = max(p['t'][-1] for p in perfiles)
        frame_times = np.linspace(0, t_end, n_frames)
        for p in perfiles:
            p['pos_i'] = np.interp(frame_times, p['t'], p['pos'])
            p['vel_i'] = np.interp(frame_times, p['t'], p['vel'])

        n_lanes = len(perfiles)
        rows = [1, 0] if n_lanes == 2 else [0]
        max_overrun = max(p.get('overrun_mm', 0.0) + (p.get('slip_mm', 0.0) or 0.0) for p in perfiles)
        x_max = length + max(max_overrun * 3, length * 0.12) + length * 0.10

        fig_t = go.Figure()

        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="#ff7f0e", dash="dot", width=2), name="Sensor Reducción"))
        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="#d62728", dash="dash", width=2), name="Ziel (Sensor Paro)"))
        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="rgba(220,50,50,0.35)", size=15, symbol='square'), name="Zona Overrun / Slip"))
        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol="triangle-down", size=12, color="black"), name="Alto Real (posición final)"))

        for idx, p in enumerate(perfiles):
            row = rows[idx]
            color = p['color']
            fig_t.add_trace(go.Scatter(x=[0, length], y=[row, row], mode='lines',
                                        line=dict(color="#e6e6e6", width=24), showlegend=False, hoverinfo='skip'))
            pos_sr = p['pos_sensor_red']
            fig_t.add_shape(type="line", x0=pos_sr, x1=pos_sr, y0=row - 0.32, y1=row + 0.32,
                             line=dict(color="#ff7f0e", width=2, dash="dot"))
            fig_t.add_shape(type="line", x0=length, x1=length, y0=row - 0.32, y1=row + 0.32,
                             line=dict(color="#d62728", width=2, dash="dash"))
            overrun_mm = p.get('overrun_mm', 0.0)
            slip_mm = p.get('slip_mm', 0.0) or 0.0
            zona_fin = length + overrun_mm + slip_mm
            if zona_fin > length:
                fig_t.add_shape(type="rect", x0=length, x1=zona_fin, y0=row - 0.26, y1=row + 0.26,
                                 fillcolor="rgba(220,50,50,0.18)", line=dict(width=0))
            pos_final_real = p['pos'][-1] + slip_mm
            fig_t.add_trace(go.Scatter(x=[pos_final_real], y=[row + 0.34], mode='markers',
                                        marker=dict(symbol="triangle-down", size=11, color="black"),
                                        showlegend=False, hoverinfo='skip'))
            info_txt = (f"<b>{p['label']}</b><br>"
                        f"v_fast={p['v_fast']:.0f} · v_slow={p['v_slow']:.0f} mm/s<br>"
                        f"accel={p['accel']:.0f} · decel={p['decel']:.0f} mm/s²<br>"
                        f"μ={p['mu']:.2f} · t_ciclo={p['t'][-1]:.2f} s")
            fig_t.add_annotation(x=0.0, y=row, xref="paper", yref="y", xanchor="left",
                                  text=info_txt, showarrow=False, align="left",
                                  font=dict(size=11, color="#333"), xshift=-18)
            if overrun_mm > 0:
                fig_t.add_annotation(x=length, y=row - 0.42, xref="x", yref="y",
                                      text=f"Overrun: +{overrun_mm:.1f} mm / +{p.get('overrun_ms', 0):.0f} ms",
                                      showarrow=False, font=dict(size=10, color="#b00000"))

        fig_t.add_annotation(x=0, y=1.16, xref="x", yref="paper", text="<b>Start</b>", showarrow=False, font=dict(size=12, color="gray"))
        fig_t.add_annotation(x=length, y=1.16, xref="x", yref="paper", text="<b>Ziel</b>", showarrow=False, font=dict(size=12, color="#444"))

        base_traces = len(fig_t.data)
        for idx, p in enumerate(perfiles):
            row = rows[idx]
            fig_t.add_trace(go.Scatter(x=[0, p['pos_i'][0]], y=[row, row], mode='lines',
                                        line=dict(color=p['color'], width=20), showlegend=False, hoverinfo='skip'))
            fig_t.add_trace(go.Scatter(x=[p['pos_i'][0]], y=[row], mode='markers',
                                        marker=dict(symbol='square', size=24, color=p['color'],
                                                     line=dict(color='black', width=2)),
                                        showlegend=False, hoverinfo='skip'))
            fig_t.add_trace(go.Scatter(x=[length * 1.02], y=[row], mode='text', textposition="middle left",
                                        text=["..."], showlegend=False, hoverinfo='skip',
                                        textfont=dict(size=11, color="#333")))

        frames = []
        for k, ft in enumerate(frame_times):
            data_k, idx_k = [], []
            for idx, p in enumerate(perfiles):
                row = rows[idx]
                pos_k, vel_k = p['pos_i'][k], p['vel_i'][k]
                pct_k = 100.0 * min(pos_k / length, 1.0)
                st_name = get_state(ft, p['t_reach_fast'], p['t_red'], p['t_reach_slow'], p['t_stop'], p['t_full'])
                icon, label_st = STATE_INFO[st_name]
                marker_color = STATE_COLOR[st_name] or p['color']
                is_slip = (st_name == "DECEL_TO_STOP") and p.get('se_desliza', False)
                g_val = p['g_por_fase'].get(st_name, 0.0)
                txt = f"{icon} <b>{label_st}</b><br>{pos_k:.0f} mm | {vel_k:.1f} mm/s ({pct_k:.0f}%)<br>G: {g_val:.3f}"
                if is_slip: txt += " ⚠️ SLIP"
                txt_color = "#e74c3c" if is_slip else "#333"
                fi = base_traces + idx * 3
                data_k.append(go.Scatter(x=[0, pos_k], y=[row, row], line=dict(color=marker_color))); idx_k.append(fi)
                data_k.append(go.Scatter(x=[pos_k], y=[row], marker=dict(color=marker_color))); idx_k.append(fi + 1)
                data_k.append(go.Scatter(x=[length * 1.02], y=[row], text=[txt], textfont=dict(color=txt_color))); idx_k.append(fi + 2)
            frames.append(go.Frame(data=data_k, traces=idx_k, name=str(k)))

        fig_t.frames = frames
        target_total_ms = float(np.clip(t_end * 1000.0, 4000.0, 16000.0))
        frame_ms = max((target_total_ms / n_frames) / speed_mult, 8.0)

        fig_t.update_layout(
            height=200 + n_lanes * 190, template="plotly_white",
            xaxis=dict(title="Posición (mm)", range=[-length * 0.02, x_max]),
            yaxis=dict(visible=False, range=[-0.7, max(rows) + 0.7]),
            margin=dict(l=230, t=70, b=60, r=40),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
            updatemenus=[dict(type="buttons", showactive=False, y=1.22, x=0.0, xanchor="left",
                buttons=[dict(label="▶ Play", method="animate",
                              args=[None, dict(frame=dict(duration=frame_ms, redraw=False), fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
                         dict(label="⏸ Pause", method="animate",
                              args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
            sliders=[dict(steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=False))], label=f"{frame_times[k]:.1f}s") for k in range(n_frames)],
                          x=0.0, len=0.96, y=-0.32)]
        )
        return fig_t

    def g_por_fase(accel, decel, g_conv):
        return {"ACCEL_FAST": accel / 9810.0, "CRUISE_FAST": 0.0, "DECEL_TO_SLOW": decel / 9810.0,
                "CRUISE_SLOW": 0.0, "DECEL_TO_STOP": g_conv, "DONE": 0.0}

    perfiles_animacion = [{
        "label": "Perfil A", "color": "#1f77b4", "t": t_a, "pos": pos_a, "vel": vel_a,
        "v_fast": st.session_state.speed_fast_a, "v_slow": st.session_state.speed_slow_a,
        "accel": st.session_state.accel_a, "decel": st.session_state.decel_a, "mu": st.session_state.mu_a,
        "pos_sensor_red": det_a['pos_sensor_red'], "t_reach_fast": det_a['t_reach_fast'], "t_red": t_red_a,
        "t_reach_slow": det_a['t_reach_slow'], "t_stop": t_stop_a, "t_full": det_a['t_fully_stopped'],
        "overrun_mm": overrun_a, "overrun_ms": overrun_time_a * 1000, "slip_mm": d_desliza_a,
        "se_desliza": desliza_a, "g_por_fase": g_por_fase(st.session_state.accel_a, st.session_state.decel_a, g_conv_a),
    }]
    if st.session_state.comparar:
        perfiles_animacion.append({
            "label": "Perfil B", "color": "#d62728", "t": t_b, "pos": pos_b, "vel": vel_b,
            "v_fast": st.session_state.speed_fast_b, "v_slow": st.session_state.speed_slow_b,
            "accel": st.session_state.accel_b, "decel": st.session_state.decel_b, "mu": st.session_state.mu_b,
            "pos_sensor_red": det_b['pos_sensor_red'], "t_reach_fast": det_b['t_reach_fast'], "t_red": t_red_b,
            "t_reach_slow": det_b['t_reach_slow'], "t_stop": t_stop_b, "t_full": det_b['t_fully_stopped'],
            "overrun_mm": overrun_b, "overrun_ms": overrun_time_b * 1000, "slip_mm": d_desliza_b,
            "se_desliza": desliza_b, "g_por_fase": g_por_fase(st.session_state.accel_b, st.session_state.decel_b, g_conv_b),
        })

    fig_track = construir_track(perfiles_animacion, st.session_state.conveyor_length, n_frames, speed_mult)
    st.plotly_chart(fig_track, use_container_width=True)

    # ==========================================
    # 6. WORK PIECE CARRIER + CARGA (EJE FIJO)
    # ==========================================
    st.markdown("---")
    st.subheader("🔬 Work Piece Carrier & Carga: Deslizamiento en Vivo")
    st.caption("El Carrier (160×25 cm, gris) va rígidamente sobre las ruedas del conveyor — nunca desliza. "
               "La Carga (120×13 cm, color) va encima sujeta solo por fricción.")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        velocidad_carrier = st.select_slider("Velocidad de reproducción", options=["0.5x", "1x", "2x", "4x"], value="1x", key="vel_carrier")
    with col_ctrl2:
        exageracion_txt = st.select_slider("Exageración visual del deslizamiento",
            options=["1x (real)", "5x", "10x", "25x", "50x"], value="10x", key="exag_slip")
    speed_mult_carrier = speed_map[velocidad_carrier]
    exag_map = {"1x (real)": 1.0, "5x": 5.0, "10x": 10.0, "25x": 25.0, "50x": 50.0}
    exageracion = exag_map[exageracion_txt]
    n_frames_carrier = 240

    def construir_carrier_load(perfiles, length, n_frames, speed_mult, exageracion,
                                carrier_alto=CARRIER_ALTO_MM, carrier_largo=CARRIER_LARGO_MM,
                                carga_alto=CARGA_ALTO_MM, carga_largo=CARGA_LARGO_MM):
        margin = (carrier_largo - carga_largo) / 2.0
        for p in perfiles:
            if p['se_desliza']:
                p['t_load_stop'] = p['t_stop'] + (p['v_slow'] / p['a_max_pieza'] if p['a_max_pieza'] > 0 else 0.0)
            else:
                p['t_load_stop'] = p['t_full']

        t_start = min(p['t_red'] for p in perfiles)
        t_end = max(max(p['t_full'], p['t_load_stop']) for p in perfiles)
        frame_times = np.linspace(t_start, t_end, n_frames)

        for p in perfiles:
            carrier_pos_i = np.interp(frame_times, p['t'], p['pos'])
            if p['se_desliza']:
                load_pos_i = carrier_pos_i.copy()
                mask = frame_times >= p['t_stop']
                dt = frame_times[mask] - p['t_stop']
                pos_load_final = length + p['dist_freno_pieza']
                t_full_load = p['v_slow'] / p['a_max_pieza'] if p['a_max_pieza'] > 0 else 0.0
                seg = length + p['v_slow'] * dt - 0.5 * p['a_max_pieza'] * dt ** 2
                seg = np.where(dt <= t_full_load, seg, pos_load_final)
                load_pos_i[mask] = seg
            else:
                load_pos_i = carrier_pos_i.copy()
            p['carrier_pos_i'] = carrier_pos_i
            p['load_pos_i'] = load_pos_i
            p['slip_i'] = np.maximum(load_pos_i - carrier_pos_i, 0.0)
            p['lf_i'] = carrier_pos_i - margin + p['slip_i'] * exageracion

        n_lanes = len(perfiles)
        lane_gap = 220.0
        lane_height = carrier_alto + carga_alto + lane_gap
        for idx, p in enumerate(perfiles):
            p['y_off'] = idx * lane_height

        left_bound = min(float(np.min(p['carrier_pos_i'])) - carrier_largo for p in perfiles)
        right_bound = max(float(np.max(p['lf_i'])) for p in perfiles)
        pad = max((right_bound - left_bound) * 0.05, 80.0)
        x_min_fixed = left_bound - pad
        x_max_fixed = right_bound + pad

        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="#ff7f0e", dash="dot", width=2), name="Sensor Reducción"))
        fig_c.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="#d62728", dash="dash", width=2), name="Ziel (Sensor Paro)"))
        fig_c.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="#95a5a6", size=15, symbol='square', line=dict(color='black', width=1)), name="Work Piece Carrier (sin slip)"))
        fig_c.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="rgba(220,50,50,0.6)", size=15, symbol='square'), name="Carga deslizando (slip activo)"))

        for p in perfiles:
            y0c = p['y_off']
            fig_c.add_shape(type="line", x0=x_min_fixed, x1=x_max_fixed, y0=y0c, y1=y0c, line=dict(color="#bbbbbb", width=2))
            fig_c.add_shape(type="line", x0=p['pos_sensor_red'], x1=p['pos_sensor_red'], y0=y0c - 40, y1=y0c + carrier_alto + carga_alto + 60, line=dict(color="#ff7f0e", width=2, dash="dot"))
            fig_c.add_shape(type="line", x0=length, x1=length, y0=y0c - 40, y1=y0c + carrier_alto + carga_alto + 60, line=dict(color="#d62728", width=2, dash="dash"))
            info_txt = (f"<b>{p['label']}</b><br>v_slow={p['v_slow']:.0f} mm/s · μ={p['mu']:.2f}<br>Slip máx real: {p['slip_i'].max():.3f} mm")
            fig_c.add_annotation(x=0.0, y=y0c + (carrier_alto + carga_alto) / 2, xref="paper", yref="y", xanchor="left",
                                  text=info_txt, showarrow=False, align="left", font=dict(size=11, color="#333"), xshift=-18)

        base_traces = len(fig_c.data)
        for p in perfiles:
            y0c = p['y_off']
            cf0 = p['carrier_pos_i'][0]
            lf0 = p['lf_i'][0]
            fig_c.add_trace(go.Scatter(x=[cf0 - carrier_largo, cf0, cf0, cf0 - carrier_largo, cf0 - carrier_largo],
                                        y=[y0c, y0c, y0c + carrier_alto, y0c + carrier_alto, y0c],
                                        mode='lines', fill='toself', fillcolor="#95a5a6",
                                        line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
            fig_c.add_trace(go.Scatter(x=[lf0 - carga_largo, lf0, lf0, lf0 - carga_largo, lf0 - carga_largo],
                                        y=[y0c + carrier_alto, y0c + carrier_alto, y0c + carrier_alto + carga_alto, y0c + carrier_alto + carga_alto, y0c + carrier_alto],
                                        mode='lines', fill='toself', fillcolor=p['color'],
                                        line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
            fig_c.add_trace(go.Scatter(x=[cf0], y=[y0c + carrier_alto + carga_alto + 35], mode='text',
                                        text=["..."], textfont=dict(size=11, color="#333"), showlegend=False, hoverinfo='skip'))

        frames = []
        for k, ft in enumerate(frame_times):
            data_k, idx_k = [], []
            for pi, p in enumerate(perfiles):
                y0c = p['y_off']
                cf = p['carrier_pos_i'][k]
                lf = p['lf_i'][k]
                slip_now = p['slip_i'][k]
                is_slip_now = ft >= p['t_stop'] and slip_now > 1e-6
                load_color = "#e74c3c" if is_slip_now else p['color']
                fi = base_traces + pi * 3
                data_k.append(go.Scatter(x=[cf - carrier_largo, cf, cf, cf - carrier_largo, cf - carrier_largo],
                                          y=[y0c, y0c, y0c + carrier_alto, y0c + carrier_alto, y0c])); idx_k.append(fi)
                data_k.append(go.Scatter(x=[lf - carga_largo, lf, lf, lf - carga_largo, lf - carga_largo],
                                          y=[y0c + carrier_alto, y0c + carrier_alto, y0c + carrier_alto + carga_alto, y0c + carrier_alto + carga_alto, y0c + carrier_alto],
                                          fillcolor=load_color)); idx_k.append(fi + 1)
                exag_note = f" (mostrado a {exageracion:.0f}x)" if exageracion > 1 and slip_now > 0 else ""
                txt = f"Carrier: {cf:.0f} mm | Slip real: {slip_now:.3f} mm ({slip_now*1000:.0f} µm){exag_note}"
                if is_slip_now: txt = "⚠️ DESLIZANDO — " + txt
                data_k.append(go.Scatter(x=[cf], y=[y0c + carrier_alto + carga_alto + 35], text=[txt],
                                          textfont=dict(color="#e74c3c" if is_slip_now else "#333"))); idx_k.append(fi + 2)
            frames.append(go.Frame(data=data_k, traces=idx_k, name=str(k)))

        fig_c.frames = frames
        target_total_ms = float(np.clip((t_end - t_start) * 1000.0, 4000.0, 14000.0))
        frame_ms = max((target_total_ms / n_frames) / speed_mult, 8.0)

        fig_c.update_layout(
            height=180 + n_lanes * (carrier_alto / 2 + carga_alto + 240), template="plotly_white",
            xaxis=dict(title="Posición (mm) — eje fijo", range=[x_min_fixed, x_max_fixed]),
            yaxis=dict(visible=False, range=[-60, n_lanes * lane_height + 60]),
            margin=dict(l=200, t=70, b=60, r=40),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
            updatemenus=[dict(type="buttons", showactive=False, y=1.15, x=0.0, xanchor="left",
                buttons=[dict(label="▶ Play", method="animate",
                              args=[None, dict(frame=dict(duration=frame_ms, redraw=False), fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
                         dict(label="⏸ Pause", method="animate",
                              args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
            sliders=[dict(steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=False))], label=f"{frame_times[k]:.2f}s") for k in range(n_frames)],
                          x=0.0, len=0.96, y=-0.30)]
        )
        return fig_c

    perfiles_carrier = [{
        "label": "Perfil A", "color": "#1f77b4", "t": t_a, "pos": pos_a,
        "v_slow": st.session_state.speed_slow_a, "mu": st.session_state.mu_a, "a_max_pieza": det_a['a_max_pieza'],
        "se_desliza": desliza_a, "dist_freno_pieza": det_a['dist_freno_pieza'], "t_stop": t_stop_a, "t_red": t_red_a,
        "pos_sensor_red": det_a['pos_sensor_red'], "t_full": det_a['t_fully_stopped'],
    }]
    if st.session_state.comparar:
        perfiles_carrier.append({
            "label": "Perfil B", "color": "#d62728", "t": t_b, "pos": pos_b,
            "v_slow": st.session_state.speed_slow_b, "mu": st.session_state.mu_b, "a_max_pieza": det_b['a_max_pieza'],
            "se_desliza": desliza_b, "dist_freno_pieza": det_b['dist_freno_pieza'], "t_stop": t_stop_b, "t_red": t_red_b,
            "pos_sensor_red": det_b['pos_sensor_red'], "t_full": det_b['t_fully_stopped'],
        })

    fig_carrier = construir_carrier_load(perfiles_carrier, st.session_state.conveyor_length, n_frames_carrier, speed_mult_carrier, exageracion)
    st.plotly_chart(fig_carrier, use_container_width=True)

    # ==========================================
    # 7. MESA GIRATORIA (TURNING TABLE)
    # ==========================================
    st.markdown("---")
    st.header("🔄 Mesa Giratoria (Turning Table)")
    st.caption(
        f"Pieza centrada en el eje de giro. R_max (deslizamiento) = Longitud/2 = {r_max_mesa:.1f} mm. "
        f"⚠️ Distinto del Radio Riel Sensores ({st.session_state.radio_sensor_mesa:.2f} mm), que solo se usa "
        f"para convertir posiciones de sensores entre grados y mm."
    )

    # ---- GRÁFICA DE VELOCIDAD ANGULAR ----
    fig_mesa_vel = go.Figure()
    fig_mesa_vel.add_trace(go.Scatter(x=t_m, y=omega_m, mode='lines', name='Angular Velocity (Mesa)',
                                       line=dict(color='#1f77b4', width=3)))

    fig_mesa_vel.add_vline(x=t_red_m, line_width=2, line_dash="dot", line_color="#ff7f0e",
                            annotation_text=f"Sensor Reducción Mesa · {t_red_m:.2f}s",
                            annotation_position="top left", annotation_font_size=10, annotation_font_color="#ff7f0e")
    fig_mesa_vel.add_vline(x=t_stop_m, line_width=2, line_dash="dash", line_color="#d62728",
                            annotation_text=f"Sensor Paro Mesa · {t_stop_m:.2f}s",
                            annotation_position="top right", annotation_font_size=10, annotation_font_color="#d62728")

    fig_mesa_vel.update_layout(
        title="Angular Velocity Profile (°/s) vs Time (s)",
        xaxis_title="Time (s)", yaxis_title="Angular Velocity (°/s)",
        height=480, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(b=90, t=90)
    )
    st.plotly_chart(fig_mesa_vel, use_container_width=True)

    if det_m['insufficient_angle']:
        st.warning(
            f"⚠️ **Mesa — Configuración inconsistente:** `RAMP_DECEL MESA` necesita "
            f"**{det_m['angle_needed_decel']:.1f}°** para bajar de SPEED_AUTO_FAST a SPEED_AUTO_SLOW, "
            f"pero `Ángulo Sensor Reducción MESA` solo tiene **{st.session_state.sensor_angle_mesa:.1f}°**."
        )

    # ---- NUEVA SECCIÓN: OVERRUN ANGULAR Y CONVERSIÓN AL RIEL DE SENSORES ----
    st.markdown("---")
    st.subheader("📏 Overrun Angular y su Conversión al Riel de Sensores (mm)")
    st.caption(
        "Los sensores de esta mesa NO giran — están fijos sobre un riel a un radio constante desde el centro "
        "de rotación. Esta sección convierte el overrun angular (causado por RAMP_STOP) a la distancia lineal "
        "equivalente sobre ese riel, para que mantenimiento pueda reposicionar sensores usando cinta métrica."
    )

    modo_conversion = st.radio(
        "Fórmula de conversión Ángulo ↔ mm",
        ["Arco (R·θ) — recomendado", "Cuerda (2R·sin(θ/2))"],
        horizontal=True, key="modo_conversion_mesa",
        help="Arco = distancia exacta si el riel sigue la curvatura a ese radio (caso típico). "
             "Cuerda = distancia recta entre dos puntos del círculo (útil si el riel es perfectamente recto)."
    )
    modo_key = "Arco (R·θ)" if modo_conversion.startswith("Arco") else "Cuerda (2R·sin(θ/2))"
    R_sensor = st.session_state.radio_sensor_mesa

    overrun_mm_riel = angle_to_mm(ang_overrun_m, R_sensor, modo_key)

    ov1, ov2, ov3 = st.columns(3)
    ov1.metric("Overrun Angular", f"{ang_overrun_m:.4f}°")
    ov2.metric(f"Overrun en el Riel (R={R_sensor:.2f}mm)", f"{overrun_mm_riel:.3f} mm")
    ov3.metric("Overrun Time", f"{overrun_time_m*1000:.0f} ms")

    st.markdown("**Memoria de cálculo — Overrun Angular (análogo directo al conveyor lineal):**")
    st.latex(L(
        f"~Delta~theta_{{~text{{overrun}}}} ~approx ~frac{{1}}{{2}} ~cdot ~omega_{{~text{{slow}}}} ~cdot t_{{~text{{stop~_real}}}} "
        f"= 0.5 ~cdot {st.session_state.speed_slow_mesa:.1f}~,°/s ~cdot {det_m['ramp_stop_real_ms']/1000.0:.3f}~,s "
        f"~approx ~mathbf{{{ang_overrun_m:.4f}°}}"
    ))
    if modo_key == "Arco (R·θ)":
        st.latex(L(
            f"s_{{~text{{mm}}}} = R_{{~text{{sensor}}}} ~cdot ~theta_{{~text{{rad}}}} "
            f"= {R_sensor:.2f} ~cdot ~frac{{{ang_overrun_m:.4f} ~cdot ~pi}}{{180}} "
            f"= ~mathbf{{{overrun_mm_riel:.3f}~,~text{{mm}}}}"
        ))
    else:
        st.latex(L(
            f"s_{{~text{{mm}}}} = 2R_{{~text{{sensor}}}} ~sin~left(~frac{{~theta_{{~text{{rad}}}}}}{{2}}~right) "
            f"= ~mathbf{{{overrun_mm_riel:.3f}~,~text{{mm}}}}"
        ))

    st.markdown("**🔧 Recomendación de compensación (para preservar la posición de paro original):**")
    trigger_angle_original = st.session_state.angle_total_mesa
    trigger_angle_compensado = trigger_angle_original - ang_overrun_m
    trigger_mm_original = angle_to_mm(trigger_angle_original, R_sensor, modo_key)
    trigger_mm_compensado = angle_to_mm(trigger_angle_compensado, R_sensor, modo_key)

    comp1, comp2 = st.columns(2)
    comp1.metric("Posición Actual del Sensor de Paro", f"{trigger_mm_original:.2f} mm",
                 help=f"Equivalente a {trigger_angle_original:.2f}° sobre el riel a R={R_sensor:.2f}mm")
    comp2.metric("Posición Compensada Recomendada", f"{trigger_mm_compensado:.2f} mm",
                 delta=f"{(trigger_mm_compensado - trigger_mm_original):.3f} mm (mover hacia atrás)",
                 delta_color="inverse",
                 help=f"Equivalente a {trigger_angle_compensado:.2f}°")

    st.info(
        f"💡 Para que la mesa termine deteniéndose exactamente en el mismo punto que tendría con "
        f"`RAMP_STOP MESA = 0`, mueve el **sensor de paro** físicamente **{abs(trigger_mm_compensado - trigger_mm_original):.3f} mm "
        f"hacia atrás** sobre el riel (de {trigger_mm_original:.2f} mm a {trigger_mm_compensado:.2f} mm, medido desde el punto de referencia angular 0°)."
    )

    with st.expander("📐 Ver Equivalencias Completas de Sensores (Grados ↔ mm en el Riel)"):
        eq1, eq2, eq3 = st.columns(3)
        eq1.metric("Ángulo Total MESA", f"{st.session_state.angle_total_mesa:.2f}°",
                   help=f"= {angle_to_mm(st.session_state.angle_total_mesa, R_sensor, modo_key):.2f} mm en el riel")
        eq2.metric("→ mm en el Riel", f"{angle_to_mm(st.session_state.angle_total_mesa, R_sensor, modo_key):.2f} mm")
        eq3.metric("Ángulo Sensor Reducción MESA", f"{st.session_state.sensor_angle_mesa:.2f}°",
                   help=f"= {angle_to_mm(st.session_state.sensor_angle_mesa, R_sensor, modo_key):.2f} mm en el riel")
        st.caption(
            f"Posición del Sensor de Reducción (medida desde el punto de paro, hacia atrás): "
            f"**{angle_to_mm(st.session_state.sensor_angle_mesa, R_sensor, modo_key):.2f} mm** sobre el riel."
        )
        st.caption(
            "Nota: si tu riel es recto y corto (como en tu diseño, con sensores separados ~150mm), la diferencia "
            "entre Arco y Cuerda es del orden de centésimas de mm — ambos métodos son válidos en la práctica."
        )

    st.markdown("---")
    st.subheader("📊 Análisis de Deslizamiento — Frenado Final (Tangencial)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Deceleración Tangencial (frenado)", f"{g_conv_m:.3f} G")
    m2.metric("Límite de Fricción (μ)", f"{st.session_state.mu_mesa:.2f} G")
    m3.metric("Estabilidad al Frenar", "🔴 SLIPPING" if desliza_m else "🟢 STABLE")
    m4.metric("Deslizamiento Relativo", f"{det_m['deslizamiento_deg_stop']:.4f}° ({d_desliza_mm_m:.3f} mm)" if desliza_m else "0.0000°")

    fs_m = det_m['factor_seguridad_stop']
    fs_m_display = "∞" if fs_m == float('inf') else f"{fs_m:.2f}x"
    st.metric("🛡️ Factor de Seguridad al Frenar (μ / g_tan)", fs_m_display)

    st.subheader("🌀 Análisis de Riesgo Centrípeto — En Crucero (Velocidad Constante)")
    st.caption(
        "Este riesgo existe SIEMPRE que la mesa gira a velocidad constante, sin necesidad de frenar. "
        "No tiene equivalente en el conveyor lineal. Usa R_max (Longitud de Pieza / 2), no el radio del riel."
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("G Centrípeta @ Velocidad Rápida", f"{det_m['g_cent_fast']:.3f} G")
    mc2.metric("Estabilidad @ Rápida", "🔴 RIESGO" if det_m['se_desliza_cruise_fast'] else "🟢 OK")
    mc3.metric("G Centrípeta @ Velocidad Lenta", f"{det_m['g_cent_slow']:.3f} G")
    mc4.metric("Estabilidad @ Lenta", "🔴 RIESGO" if det_m['se_desliza_cruise_slow'] else "🟢 OK")

    if det_m['se_desliza_cruise_fast'] or det_m['se_desliza_cruise_slow']:
        st.error(
            "🔴 **Riesgo de deslizamiento en crucero detectado.** La fuerza centrípeta a esta velocidad angular "
            "y radio supera la fricción disponible."
        )
    else:
        st.success("🟢 La fuerza centrípeta en ambas velocidades de crucero está dentro del límite de fricción.")

    with st.expander("🔍 Ver Desglose de Cálculo — Mesa Giratoria"):
        st.markdown("### 🧮 Memoria de Cálculo (Valores Simulados)")

        st.markdown("**Paso 1: Tiempo de Rampa de Paro Efectivo**")
        st.latex(L(
            f"t_{{~text{{stop~_real}}}} = ~max({st.session_state.ramp_stop_mesa:.1f}~,~text{{ms}}, 20.0~,~text{{ms}}) "
            f"= {det_m['ramp_stop_real_ms']:.1f}~,~text{{ms}}"
        ))

        st.markdown("**Paso 2: Deceleración Angular Final y su Componente Tangencial en R_max**")
        st.latex(L(
            f"~alpha_{{~text{{stop}}}} = ~frac{{{st.session_state.speed_slow_mesa:.1f}~,°/s}}"
            f"{{{det_m['ramp_stop_real_ms']/1000.0:.3f}~,s}} = {det_m['alpha_stop']:.2f}~,°/s^2"
        ))
        st.latex(L(
            f"a_{{~text{{tan~_stop}}}} = ~alpha_{{~text{{stop}}}} ~cdot R_{{~text{{max}}}} "
            f"= {det_m['a_tan_stop']:.2f}~,~text{{mm/s}}^2"
        ))
        st.latex(L(
            f"g_{{~text{{conv}}}} = ~frac{{{det_m['a_tan_stop']:.2f}}}{{9810}} = ~mathbf{{{g_conv_m:.3f}~,G}}"
        ))

        st.markdown("**Paso 3: Componente Centrípeta en Crucero**")
        st.latex(L(r"a_{~text{cent}} = ~omega^2 ~cdot R_{~text{max}} ~quad ~text{(en radianes)}"))
        st.latex(L(
            f"g_{{~text{{cent,fast}}}} = ~mathbf{{{det_m['g_cent_fast']:.3f}~,G}} ~quad "
            f"g_{{~text{{cent,slow}}}} = ~mathbf{{{det_m['g_cent_slow']:.3f}~,G}}"
        ))

        st.markdown("**Paso 4: Criterio de Deslizamiento al Frenar**")
        if desliza_m:
            st.error(f"🔴 **SLIP AL FRENAR:** g_conv ({g_conv_m:.3f} G) > μ ({st.session_state.mu_mesa:.2f} G)")
        else:
            st.success(f"🟢 **ESTABLE AL FRENAR:** g_conv ({g_conv_m:.3f} G) ≤ μ ({st.session_state.mu_mesa:.2f} G)")

    st.markdown("---")
    st.subheader("🎯 Posicionamiento y Tiempo de Ciclo — Mesa")
    mp1, mp2, mp3, mp4 = st.columns(4)
    mp1.metric("Overrun Angular", f"{ang_overrun_m:.3f}°")
    mp2.metric("Overrun Time", f"{overrun_time_m*1000:.0f} ms")
    mp3.metric("Posición Final Pieza", f"{(theta_m[-1] + det_m['deslizamiento_deg_stop']):.3f}°")
    mp4.metric("Tiempo Total de Ciclo", f"{t_m[-1]:.2f} s")

    st.markdown("---")
    st.subheader("🎬 Animación: Vista Superior de la Mesa Girando")
    st.caption(
        "Vista desde arriba. El rectángulo gris representa la mesa/carrier girando (sin slip). "
        "La pieza (color) se muestra centrada; se pone roja durante el frenado final si desliza tangencialmente."
    )

    vel_mesa_anim = st.select_slider("Velocidad de reproducción (Mesa)", options=["0.5x", "1x", "2x", "4x"], value="1x", key="vel_mesa")
    exag_mesa_txt = st.select_slider("Exageración visual del deslizamiento angular",
                                      options=["1x (real)", "10x", "50x", "200x"], value="50x", key="exag_mesa")
    speed_mult_mesa = speed_map[vel_mesa_anim]
    exag_mesa_map = {"1x (real)": 1.0, "10x": 10.0, "50x": 50.0, "200x": 200.0}
    exag_mesa = exag_mesa_map[exag_mesa_txt]
    n_frames_mesa = 220

    def construir_mesa_animacion(t_arr, theta_arr, omega_arr, t_stop, t_full, angle_stop,
                                  se_desliza, deslizamiento_deg, alpha_max_pieza_deg,
                                  omega_slow, r_max, n_frames, speed_mult, exageracion,
                                  ancho_pieza=PIEZA_MESA_ANCHO_MM):
        frame_times = np.linspace(0, t_arr[-1], n_frames)
        theta_table_i = np.interp(frame_times, t_arr, theta_arr)

        theta_load_i = theta_table_i.copy()
        if se_desliza:
            mask = frame_times >= t_stop
            dtl = frame_times[mask] - t_stop
            t_full_load = omega_slow / alpha_max_pieza_deg if alpha_max_pieza_deg > 0 else 0.0
            seg = angle_stop + omega_slow * dtl - 0.5 * alpha_max_pieza_deg * dtl ** 2
            theta_final_load = angle_stop + deslizamiento_deg
            seg = np.where(dtl <= t_full_load, seg, theta_final_load)
            theta_load_i[mask] = seg

        slip_deg_i = np.maximum(theta_load_i - theta_table_i, 0.0)
        theta_load_visual_i = theta_table_i + slip_deg_i * exageracion

        table_radius = r_max * 1.25
        L_pieza = r_max * 2.0
        W_pieza = ancho_pieza

        def rect_corners(theta_deg, largo, ancho):
            th_rad = np.radians(theta_deg)
            hl, hw = largo / 2.0, ancho / 2.0
            local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]])
            c, s = np.cos(th_rad), np.sin(th_rad)
            rot = np.array([[c, -s], [s, c]])
            rotated = local @ rot.T
            return rotated[:, 0], rotated[:, 1]

        fig_m = go.Figure()

        fig_m.add_shape(type="circle", x0=-table_radius, x1=table_radius, y0=-table_radius, y1=table_radius,
                         line=dict(color="#999999", width=2), fillcolor="rgba(230,230,230,0.4)")

        fig_m.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="#95a5a6", width=3), name="Mesa (rígida, sin slip)"))
        fig_m.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="rgba(220,50,50,0.8)", width=3), name="Pieza deslizando (slip activo)"))

        x0m, y0m = rect_corners(theta_table_i[0], L_pieza * 0.06, W_pieza * 1.15)
        fig_m.add_trace(go.Scatter(x=x0m, y=y0m, mode='lines', fill='toself', fillcolor="#95a5a6",
                                    line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))

        x0p, y0p = rect_corners(theta_load_visual_i[0], L_pieza, W_pieza)
        fig_m.add_trace(go.Scatter(x=x0p, y=y0p, mode='lines', fill='toself', fillcolor="#1f77b4",
                                    line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))

        fig_m.add_trace(go.Scatter(x=[0], y=[table_radius * 1.18], mode='text', text=["..."],
                                    textfont=dict(size=12, color="#333"), showlegend=False, hoverinfo='skip'))

        idx_gray = len(fig_m.data) - 3
        idx_piece = len(fig_m.data) - 2
        idx_text = len(fig_m.data) - 1

        frames = []
        for k, ft in enumerate(frame_times):
            xm, ym = rect_corners(theta_table_i[k], L_pieza * 0.06, W_pieza * 1.15)
            xp, yp = rect_corners(theta_load_visual_i[k], L_pieza, W_pieza)
            slip_now = slip_deg_i[k]
            is_slip_now = ft >= t_stop and slip_now > 1e-9
            color_now = "#e74c3c" if is_slip_now else "#1f77b4"
            omega_now = np.interp(ft, t_arr, omega_arr)
            exag_note = f" (x{exageracion:.0f})" if exageracion > 1 and slip_now > 0 else ""
            txt = f"θ_mesa={theta_table_i[k]:.2f}° | ω={omega_now:.1f}°/s"
            if is_slip_now:
                txt = f"⚠️ DESLIZANDO — Δθ={slip_now:.4f}°{exag_note} | " + txt
            frames.append(go.Frame(
                data=[go.Scatter(x=xm, y=ym), go.Scatter(x=xp, y=yp, fillcolor=color_now), go.Scatter(x=[0], y=[table_radius * 1.18], text=[txt])],
                traces=[idx_gray, idx_piece, idx_text],
                name=str(k)
            ))

        fig_m.frames = frames
        target_total_ms = float(np.clip(t_arr[-1] * 1000.0, 4000.0, 14000.0))
        frame_ms = max((target_total_ms / n_frames) / speed_mult, 8.0)

        fig_m.update_layout(
            height=560, template="plotly_white",
            xaxis=dict(range=[-table_radius * 1.3, table_radius * 1.3], scaleanchor="y", title="mm"),
            yaxis=dict(range=[-table_radius * 1.3, table_radius * 1.3], title="mm"),
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", font=dict(size=10)),
            margin=dict(l=40, t=70, b=60, r=40),
            updatemenus=[dict(type="buttons", showactive=False, y=1.1, x=0.0, xanchor="left",
                buttons=[dict(label="▶ Play", method="animate",
                              args=[None, dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
                         dict(label="⏸ Pause", method="animate",
                              args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
            sliders=[dict(steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=f"{frame_times[k]:.2f}s") for k in range(n_frames)],
                          x=0.0, len=0.96, y=-0.22)]
        )
        return fig_m

    fig_mesa = construir_mesa_animacion(
        t_m, theta_m, omega_m, t_stop_m, det_m['t_fully_stopped'], det_m['angle_stop'],
        desliza_m, det_m['deslizamiento_deg_stop'], det_m['alpha_max_pieza_deg'],
        st.session_state.speed_slow_mesa, r_max_mesa, n_frames_mesa, speed_mult_mesa, exag_mesa
    )
    st.plotly_chart(fig_mesa, use_container_width=True)


# ==========================================
# TAB 2: MATHEMATICAL & PHYSICAL BACKGROUND
# ==========================================
with tab_math:
    st.header("📐 Physics Engine & Kinematic Formulas")
    st.markdown(
        "This section details the analytical models used to simulate conveyor dynamics, "
        "deceleration forces, friction boundaries, and inertial displacement of transported parts."
    )

    st.subheader("1. Mechanical Elasticity Floor")
    st.latex(L(r"t_{~text{stop~_real}} = ~max~left(t_{~text{ramp~_stop}}, 20.0~,~text{ms}~right)"))

    st.subheader("2. Conveyor Stop Deceleration")
    st.latex(L(r"a_{~text{stop}} = ~frac{v_{~text{slow}}}{t_{~text{stop~_real}}}"))
    st.latex(L(r"g_{~text{conv}} = ~frac{a_{~text{stop}}}{9810}"))

    st.subheader("3. Static Friction Threshold & Slip Determination")
    st.latex(L(r"a_{~text{max~_piece}} = ~mu ~cdot g = ~mu ~cdot 9810~,~text{mm/s}^2"))

    st.subheader("4. Relative Part Slip Estimation")
    st.latex(L(r"d_{~text{piece}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{max~_piece}}}"))
    st.latex(L(r"d_{~text{conveyor}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{stop}}}"))
    st.latex(L(r"~Delta d = d_{~text{piece}} - d_{~text{conveyor}}"))

    st.subheader("5. Safety Factor & Minimum Required Friction")
    st.latex(L(r"~text{Factor de Seguridad} = ~frac{~mu}{g_{~text{conv}}}"))

    st.subheader("6. Carrier vs. Load: Two-Body Slip Model")
    st.latex(L(r"x_{~text{carrier}}(t) = x_{~text{conveyor}}(t)"))

    st.subheader("7. Sensor Distance Validation")
    st.latex(L(r"d_{~text{needed}} = ~frac{v_{~text{fast}}^2 - v_{~text{slow}}^2}{2 ~cdot ~text{RAMP~_DECEL}}"))

    st.subheader("8. Turning Table: Angular-to-Linear Analogy")
    st.latex(L(r"R_{~text{max}} = ~frac{~text{Longitud Total de Pieza}}{2}"))

    st.subheader("9. Turning Table: Tangential Component (Braking)")
    st.latex(L(r"~alpha_{~text{stop}} = ~frac{~omega_{~text{slow}}}{t_{~text{stop~_real}}} ~quad [°/s^2]"))
    st.latex(L(r"a_{~text{tan~_stop}} = ~alpha_{~text{stop,rad}} ~cdot R_{~text{max}} ~quad [~text{mm/s}^2]"))
    st.latex(L(r"g_{~text{conv}} = ~frac{a_{~text{tan~_stop}}}{9810}"))

    st.subheader("10. Turning Table: Centripetal Component (Constant Speed Cruise)")
    st.latex(L(r"a_{~text{cent}} = ~omega_{~text{rad}}^2 ~cdot R_{~text{max}} ~quad [~text{mm/s}^2]"))
    st.latex(L(r"g_{~text{cent}} = ~frac{a_{~text{cent}}}{9810}"))

    st.subheader("11. Scope Limitation: Free-Sliding Dynamics on a Rotating Frame")
    st.markdown(L(
        "A fully rigorous simulation of an object sliding freely on a rotating platform requires solving motion "
        "in a non-inertial rotating reference frame, introducing Coriolis and Euler pseudo-forces. This simulator "
        "intentionally simplifies that scenario: tangential slip during final braking is animated; centripetal "
        "risk during constant-speed cruise is reported as a numeric indicator only."
    ))

    st.subheader("12. Sensor Rail: Angular-to-Linear Conversion for Maintenance")
    st.markdown(L(
        "On this turning table, position sensors are **stationary**, mounted on a fixed rail at a constant "
        "radius $R_{~text{sensor}}$ from the rotation axis (measured directly on the machine, e.g. 545.25 mm). "
        "A flag/dog attached to the rotating table (also at that radius) trips the sensors as it sweeps past. "
        "Maintenance measures and adjusts these sensors linearly (mm) with a tape measure, not with a protractor, "
        "so the app provides an exact conversion between the angular quantities used by the physics engine and "
        "the equivalent linear distance on that rail."
    ))
    st.latex(L(r"s_{~text{arc}} = R_{~text{sensor}} ~cdot ~theta_{~text{rad}} ~quad ~text{(exact, if rail follows the circle)}"))
    st.latex(L(r"s_{~text{chord}} = 2R_{~text{sensor}} ~sin~left(~frac{~theta_{~text{rad}}}{2}~right) ~quad ~text{(exact straight-line distance)}"))
    st.markdown(L(
        "**Compensating for RAMP_STOP overrun:** since the angular overrun $~Delta~theta_{~text{overrun}}$ is added "
        "*after* the stop sensor trips (independent of where it's mounted), moving the stop sensor's trigger "
        "position backward by the arc-length equivalent of $~Delta~theta_{~text{overrun}}$ exactly restores the "
        "original final resting position:"
    ))
    st.latex(L(
        r"~text{Nueva posición sensor} = ~text{Posición original} - R_{~text{sensor}} ~cdot ~Delta~theta_{~text{overrun,rad}}"
    ))
    st.markdown(L(
        "Note $R_{~text{sensor}}$ (the sensor rail radius, a fixed machine geometry constant) is **independent** "
        "of $R_{~text{max}}$ (half the piece length, used only for the friction-slip risk analysis) — they serve "
        "entirely different purposes and must not be confused."
    ))
