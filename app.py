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
# CONSTANTE MECÁNICA GLOBAL
# ==========================================
T_MIN_MECANICO_MS = 20.0
T_MIN_MECANICO_S = T_MIN_MECANICO_MS / 1000.0

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
defaults = {
    "conveyor_length": 3000.0,
    "speed_fast_a": 300.0, "speed_slow_a": 100.0, "accel_a": 300.0, "decel_a": 300.0,
    "sensor_distance_a": 150.0, "ramp_stop_a": 2000.0, "mu_a": 0.28,
    "comparar": False,
    "speed_fast_b": 450.0, "speed_slow_b": 120.0, "accel_b": 400.0, "decel_b": 300.0,
    "sensor_distance_b": 150.0, "ramp_stop_b": 0.0, "mu_b": 0.28,
    "speed_fast_mesa": 45.0, "speed_slow_mesa": 10.0, "accel_mesa": 90.0, "decel_mesa": 90.0,
    "ramp_stop_mesa": 400.0, "mu_mesa": 0.28,
    "angle_total_mesa": 90.0, "pieza_longitud_mesa": 2110.0,
    "radio_sensor_mesa": 545.25,
    "sensor_distance_mm_mesa": 150.0,
    "modo_conversion_mesa": "Arco (R·θ) — recomendado",
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
st.sidebar.number_input(
    "RAMP_STOP A (mm/s²)", value=st.session_state.ramp_stop_a, step=100.0, min_value=0.0, key="ramp_stop_a",
    help="Deceleración directamente comandada por el PLC para el frenado final — mismo tipo de parámetro que "
         "RAMP_ACCEL / RAMP_DECEL (NO es un tiempo). Un piso mecánico (backlash de cadena, flexión de chasis) "
         "limita la deceleración máxima físicamente alcanzable a v_slow / 20ms. Si se deja en 0, se asume que "
         "el PLC no limita el frenado y el sistema se detiene tan rápido como mecánicamente es posible."
)
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
    st.sidebar.number_input(
        "RAMP_STOP B (mm/s²)", value=st.session_state.ramp_stop_b, step=100.0, min_value=0.0, key="ramp_stop_b",
        help="Igual que RAMP_STOP A — deceleración directa (mm/s²), no un tiempo. 0 = usar el máximo mecánico."
    )
    st.sidebar.number_input("Coeficiente Fricción μ B", value=st.session_state.mu_b, step=0.01, min_value=0.01, max_value=1.0, key="mu_b")

st.sidebar.markdown("---")
st.sidebar.header("🔄 Mesa Giratoria (Turning Table)")
st.sidebar.number_input("SPEED_AUTO_FAST MESA (°/s)", value=st.session_state.speed_fast_mesa, step=5.0, key="speed_fast_mesa")
st.sidebar.number_input("SPEED_AUTO_SLOW MESA (°/s)", value=st.session_state.speed_slow_mesa, step=1.0, key="speed_slow_mesa")
st.sidebar.number_input("RAMP_ACCEL MESA (°/s²)", value=st.session_state.accel_mesa, step=10.0, key="accel_mesa")
st.sidebar.number_input("RAMP_DECEL MESA (°/s²)", value=st.session_state.decel_mesa, step=10.0, key="decel_mesa")
st.sidebar.number_input("Radio Riel Sensores MESA (mm)", value=st.session_state.radio_sensor_mesa, step=5.0, min_value=1.0, key="radio_sensor_mesa", help="Radio físico donde están montados los sensores de reducción/paro sobre el riel — distinto del radio de la pieza")

st.sidebar.number_input(
    "Distancia Sensor Reducción MESA (mm)",
    value=st.session_state.sensor_distance_mm_mesa, step=5.0, min_value=0.0,
    key="sensor_distance_mm_mesa",
    help="Separación física, medida sobre el riel, entre el Sensor de Reducción y el Sensor de Paro. "
         "Ambos sensores están ubicados cerca del final del recorrido angular total."
)
_modo_actual = st.session_state.get("modo_conversion_mesa", "Arco (R·θ) — recomendado")
_sensor_angle_mesa_calc = mm_to_angle(
    st.session_state.sensor_distance_mm_mesa, st.session_state.radio_sensor_mesa, _modo_actual
)
st.sidebar.caption(f"↳ Equivalente angular: **{_sensor_angle_mesa_calc:.3f}°** (modo: {_modo_actual.split(' —')[0]})")

st.sidebar.number_input(
    "RAMP_STOP MESA (°/s²)", value=st.session_state.ramp_stop_mesa, step=25.0, min_value=0.0, key="ramp_stop_mesa",
    help="Deceleración angular directamente comandada por el PLC para el frenado final — mismo tipo de "
         "parámetro que RAMP_ACCEL/RAMP_DECEL MESA (NO es un tiempo). Un piso mecánico limita la deceleración "
         "angular máxima físicamente alcanzable a ω_slow / 20ms. Si se deja en 0, se asume el máximo mecánico."
)
st.sidebar.number_input("Coeficiente Fricción μ MESA", value=st.session_state.mu_mesa, step=0.01, min_value=0.01, max_value=1.0, key="mu_mesa")
st.sidebar.number_input("Ángulo Total de Giro MESA (°)", value=st.session_state.angle_total_mesa, step=15.0, key="angle_total_mesa", help="Equivalente angular de 'Largo Total Conveyor' — abierto")
st.sidebar.number_input("Longitud Total de Pieza MESA (mm)", value=st.session_state.pieza_longitud_mesa, step=10.0, min_value=1.0, key="pieza_longitud_mesa", help="La pieza se posiciona centrada en el eje de giro. R_max = Longitud / 2 (usado SOLO para el análisis de deslizamiento)")


# ==========================================
# REALISTIC KINEMATIC CALCULATION ENGINE (CONVEYOR LINEAL)
# ==========================================
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist, ramp_stop_accel, mu):
    dt = 0.001
    t_max = 30.0
    steps = int(t_max / dt)
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)

    p = 0.0
    v = 0.0
    pos_sensor_red = length - s_dist   # Sensor de Reducción, medido hacia atrás desde el final
    pos_stop = length                   # Sensor de Paro, al final absoluto
    state = "ACCEL_FAST"

    t_reach_fast = 0.0
    t_sensor_red = 0.0
    t_reach_slow = 0.0
    t_sensor_stop = 0.0
    t_fully_stopped = 0.0

    # ---- RAMP_STOP es ahora una ACELERACIÓN directa (mm/s²), no un tiempo ----
    a_mechanical_max = (v_slow / T_MIN_MECANICO_S) if v_slow > 0 else 0.0

    if ramp_stop_accel is None or ramp_stop_accel <= 0:
        a_stop_conveyor = a_mechanical_max
        mechanically_capped = False
    else:
        a_stop_conveyor = min(ramp_stop_accel, a_mechanical_max) if a_mechanical_max > 0 else ramp_stop_accel
        mechanically_capped = (a_mechanical_max > 0) and (ramp_stop_accel > a_mechanical_max)

    a_stop_conveyor = max(a_stop_conveyor, 1e-6)
    ramp_stop_real_ms = (v_slow / a_stop_conveyor) * 1000.0 if a_stop_conveyor > 0 else 0.0
    g_conveyor = a_stop_conveyor / 9810.0

    g_max_pieza = mu
    a_max_pieza = mu * 9810.0

    if v_fast > v_slow and decel > 0:
        dist_needed_decel = (v_fast ** 2 - v_slow ** 2) / (2.0 * decel)
    else:
        dist_needed_decel = 0.0
    insufficient_distance = dist_needed_decel > s_dist

    g_accel = accel / 9810.0
    g_decel = decel / 9810.0

    se_desliza_accel = g_accel > mu
    se_desliza_decel = g_decel > mu
    se_desliza_stop_fase = g_conveyor > mu

    fases_g = {
        "ACCEL_FAST (arranque)": g_accel,
        "DECEL_TO_SLOW (frenado a creep)": g_decel,
        "DECEL_TO_STOP (frenado final)": g_conveyor,
    }
    peor_fase_label = max(fases_g, key=fases_g.get)
    g_max_fase = fases_g[peor_fase_label]
    se_desliza_alguna_fase = g_max_fase > mu
    factor_seguridad_fase = (mu / g_max_fase) if g_max_fase > 0 else float('inf')

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
        "a_mechanical_max": a_mechanical_max,
        "mechanically_capped": mechanically_capped,
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
        "g_accel": g_accel,
        "g_decel": g_decel,
        "se_desliza_accel": se_desliza_accel,
        "se_desliza_decel": se_desliza_decel,
        "se_desliza_stop_fase": se_desliza_stop_fase,
        "fases_g": fases_g,
        "peor_fase_label": peor_fase_label,
        "g_max_fase": g_max_fase,
        "se_desliza_alguna_fase": se_desliza_alguna_fase,
        "factor_seguridad_fase": factor_seguridad_fase,
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
                          angle_total, angle_sensor_dist, ramp_stop_alpha, mu, r_max):
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

    # ---- RAMP_STOP MESA es ahora una ACELERACIÓN ANGULAR directa (°/s²), no un tiempo ----
    alpha_mechanical_max = (omega_slow / T_MIN_MECANICO_S) if omega_slow > 0 else 0.0

    if ramp_stop_alpha is None or ramp_stop_alpha <= 0:
        alpha_stop = alpha_mechanical_max
        mechanically_capped_mesa = False
    else:
        alpha_stop = min(ramp_stop_alpha, alpha_mechanical_max) if alpha_mechanical_max > 0 else ramp_stop_alpha
        mechanically_capped_mesa = (alpha_mechanical_max > 0) and (ramp_stop_alpha > alpha_mechanical_max)

    alpha_stop = max(alpha_stop, 1e-6)
    ramp_stop_real_ms = (omega_slow / alpha_stop) * 1000.0 if alpha_stop > 0 else 0.0

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

    v_fast_lineal = omega_fast_rad * r_max
    v_slow_lineal = omega_slow_rad * r_max
    alpha_accel_rad = np.radians(alpha_accel)
    alpha_decel_rad = np.radians(alpha_decel)
    a_accel_lineal = alpha_accel_rad * r_max
    a_decel_lineal = alpha_decel_rad * r_max

    a_res_fin_accel = np.sqrt(a_accel_lineal ** 2 + a_cent_fast ** 2)
    a_res_ini_decel = np.sqrt(a_decel_lineal ** 2 + a_cent_fast ** 2)
    a_res_fin_decel = np.sqrt(a_decel_lineal ** 2 + a_cent_slow ** 2)
    a_res_ini_stop = np.sqrt(a_tan_stop ** 2 + a_cent_slow ** 2)

    g_res_fin_accel = a_res_fin_accel / 9810.0
    g_res_ini_decel = a_res_ini_decel / 9810.0
    g_res_fin_decel = a_res_fin_decel / 9810.0
    g_res_ini_stop = a_res_ini_stop / 9810.0

    resultantes = {
        "Fin de ACCEL_FAST (ω=ω_fast, tan=accel)": g_res_fin_accel,
        "Inicio de DECEL_TO_SLOW (ω=ω_fast, tan=decel)": g_res_ini_decel,
        "Fin de DECEL_TO_SLOW (ω=ω_slow, tan=decel)": g_res_fin_decel,
        "Inicio de DECEL_TO_STOP (ω=ω_slow, tan=stop)": g_res_ini_stop,
    }
    peor_caso_label = max(resultantes, key=resultantes.get)
    g_res_max = resultantes[peor_caso_label]
    se_desliza_resultante = g_res_max > mu

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
    factor_seguridad_resultante = (mu / g_res_max) if g_res_max > 0 else float('inf')

    details = {
        "ramp_stop_real_ms": ramp_stop_real_ms,
        "alpha_mechanical_max": alpha_mechanical_max,
        "mechanically_capped": mechanically_capped_mesa,
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
        "v_fast_lineal": v_fast_lineal,
        "a_accel_lineal": a_accel_lineal,
        "a_decel_lineal": a_decel_lineal,
        "resultantes": resultantes,
        "peor_caso_label": peor_caso_label,
        "g_res_max": g_res_max,
        "se_desliza_resultante": se_desliza_resultante,
        "factor_seguridad_resultante": factor_seguridad_resultante,
        "g_res_fin_accel": g_res_fin_accel,
        "g_res_ini_decel": g_res_ini_decel,
        "g_res_fin_decel": g_res_fin_decel,
        "g_res_ini_stop": g_res_ini_stop,
    }

    return (t, theta, omega, t_sensor_red, t_sensor_stop, angle_overrun, overrun_time,
            g_conv_mesa, se_desliza_stop, deslizamiento_mm_stop, details)


r_max_mesa = st.session_state.pieza_longitud_mesa / 2.0

_modo_key_calc = "Arco (R·θ)" if st.session_state.get("modo_conversion_mesa", "Arco (R·θ) — recomendado").startswith("Arco") else "Cuerda (2R·sin(θ/2))"
sensor_angle_mesa_actual = mm_to_angle(
    st.session_state.sensor_distance_mm_mesa, st.session_state.radio_sensor_mesa, _modo_key_calc
)

(t_m, theta_m, omega_m, t_red_m, t_stop_m, ang_overrun_m, overrun_time_m,
 g_conv_m, desliza_m, d_desliza_mm_m, det_m) = calcular_perfil_mesa(
    st.session_state.speed_fast_mesa, st.session_state.speed_slow_mesa,
    st.session_state.accel_mesa, st.session_state.decel_mesa,
    st.session_state.angle_total_mesa, sensor_angle_mesa_actual,
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

    if det_a['mechanically_capped']:
        st.info(
            f"ℹ️ **Perfil A:** `RAMP_STOP A` configurado ({st.session_state.ramp_stop_a:.1f} mm/s²) excede el "
            f"máximo mecánico alcanzable ({det_a['a_mechanical_max']:.1f} mm/s², limitado por el piso de "
            f"{T_MIN_MECANICO_MS:.0f} ms). Se aplicó el límite mecánico: deceleración real = "
            f"**{det_a['a_stop_conveyor']:.1f} mm/s²** (≈{det_a['ramp_stop_real_ms']:.1f} ms)."
        )
    if st.session_state.comparar and det_b['mechanically_capped']:
        st.info(
            f"ℹ️ **Perfil B:** `RAMP_STOP B` configurado ({st.session_state.ramp_stop_b:.1f} mm/s²) excede el "
            f"máximo mecánico alcanzable ({det_b['a_mechanical_max']:.1f} mm/s²). Se aplicó el límite mecánico: "
            f"deceleración real = **{det_b['a_stop_conveyor']:.1f} mm/s²** (≈{det_b['ramp_stop_real_ms']:.1f} ms)."
        )

    # ---- GRÁFICA DE VELOCIDAD (LINEAL) ----
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

    # ---- SLIP ANALYSIS (frenado final) ----
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

        st.markdown("**Step 1: Mechanical Maximum Deceleration (Floor Limit)**")
        st.latex(L(
            f"a_{{~text{{mech,max}}}} = ~frac{{v_{{~text{{slow}}}}}}{{20~,~text{{ms}}}} = "
            f"~frac{{{st.session_state.speed_slow_a:.1f}~,~text{{mm/s}}}}{{0.020~,~text{{s}}}} = "
            f"~mathbf{{{det_a['a_mechanical_max']:.1f}~,~text{{mm/s}}^2}}"
        ))

        st.markdown("**Step 2: Effective Stop Deceleration (RAMP_STOP capped by mechanical limit)**")
        st.latex(L(
            f"a_{{~text{{stop}}}} = ~min({st.session_state.ramp_stop_a:.1f}~,~text{{mm/s}}^2, "
            f"{det_a['a_mechanical_max']:.1f}~,~text{{mm/s}}^2) = ~mathbf{{{det_a['a_stop_conveyor']:.1f}~,~text{{mm/s}}^2}} "
            f"~quad (t~approx{det_a['ramp_stop_real_ms']:.1f}~,~text{{ms}})"
        ))
        st.latex(L(
            f"g_{{~text{{conv}}}} = ~frac{{{det_a['a_stop_conveyor']:.1f}~,~text{{mm/s}}^2}}"
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

    # ==========================================
    # ACELERACIÓN EN GS POR TRANSICIÓN DE ESTADOS (PROFILE A)
    # ==========================================
    st.markdown("---")
    st.subheader("⚠️ Aceleración en Gs por Transición de Estados (Profile A)")
    st.caption(
        "El conveyor lineal tiene tres transiciones de velocidad donde la pieza experimenta aceleración: "
        "el arranque (ACCEL_FAST), el frenado a velocidad de creep (DECEL_TO_SLOW), y el frenado final "
        "(DECEL_TO_STOP). Cada una se evalúa por separado contra μ — el criterio de posicionamiento/overrun "
        "existente sigue basado exclusivamente en el frenado final, sin cambios."
    )

    fa1, fa2, fa3 = st.columns(3)
    fa1.metric("ACCEL_FAST (arranque)", f"{det_a['g_accel']:.3f} G",
               delta="🔴 SLIP" if det_a['se_desliza_accel'] else "🟢 OK", delta_color="off")
    fa2.metric("DECEL_TO_SLOW (frenado a creep)", f"{det_a['g_decel']:.3f} G",
               delta="🔴 SLIP" if det_a['se_desliza_decel'] else "🟢 OK", delta_color="off")
    fa3.metric("DECEL_TO_STOP (frenado final)", f"{g_conv_a:.3f} G",
               delta="🔴 SLIP" if det_a['se_desliza_stop_fase'] else "🟢 OK", delta_color="off")

    fs_fase = det_a['factor_seguridad_fase']
    fs_fase_display = "∞" if fs_fase == float('inf') else f"{fs_fase:.2f}x"
    fb1, fb2 = st.columns(2)
    fb1.metric("🔺 Fase Más Crítica", det_a['peor_fase_label'].split(" (")[0],
               help=det_a['peor_fase_label'])
    fb2.metric("🛡️ Factor de Seguridad de la Fase Más Crítica", fs_fase_display)

    if det_a['se_desliza_alguna_fase'] and not det_a['se_desliza_stop_fase']:
        st.warning(
            f"⚠️ **Diferencia detectada:** el frenado final está STABLE, pero **{det_a['peor_fase_label']}** "
            f"presenta {det_a['g_max_fase']:.3f} G, que **supera** μ ({st.session_state.mu_a:.2f} G). "
            f"La pieza podría deslizar en esa fase aunque el criterio de overrun/posicionamiento actual "
            f"(basado solo en el frenado final) no lo capture."
        )
    elif det_a['se_desliza_alguna_fase']:
        st.error(f"🔴 Al menos una fase supera μ — la más crítica es **{det_a['peor_fase_label']}**.")
    else:
        st.success("🟢 Las tres transiciones de velocidad están dentro del límite de fricción.")

    with st.expander("🔍 Ver Memoria de Cálculo — Aceleración por Fase (Profile A)"):
        st.latex(L(
            f"g_{{~text{{accel}}}} = ~frac{{~text{{RAMP~_ACCEL}}}}{{9810}} = "
            f"~frac{{{st.session_state.accel_a:.1f}}}{{9810}} = ~mathbf{{{det_a['g_accel']:.3f}~,G}}"
        ))
        st.latex(L(
            f"g_{{~text{{decel}}}} = ~frac{{~text{{RAMP~_DECEL}}}}{{9810}} = "
            f"~frac{{{st.session_state.decel_a:.1f}}}{{9810}} = ~mathbf{{{det_a['g_decel']:.3f}~,G}}"
        ))
        st.latex(L(
            f"g_{{~text{{stop}}}} = ~frac{{a_{{~text{{stop}}}}}}{{9810}} = ~mathbf{{{g_conv_a:.3f}~,G}} ~quad "
            f"~text{{(ya calculado en el frenado final, con el piso mecánico aplicado si corresponde)}}"
        ))

    if st.session_state.comparar:
        st.markdown("**⚖️ Comparación con Perfil B:**")
        fa1b, fa2b, fa3b = st.columns(3)
        fa1b.metric("ACCEL_FAST B", f"{det_b['g_accel']:.3f} G",
                    delta=f"{(det_b['g_accel'] - det_a['g_accel']):.3f} G", delta_color="inverse")
        fa2b.metric("DECEL_TO_SLOW B", f"{det_b['g_decel']:.3f} G",
                    delta=f"{(det_b['g_decel'] - det_a['g_decel']):.3f} G", delta_color="inverse")
        fa3b.metric("DECEL_TO_STOP B", f"{g_conv_b:.3f} G",
                    delta=f"{(g_conv_b - g_conv_a):.3f} G", delta_color="inverse")
        st.caption(f"Fase más crítica en B: **{det_b['peor_fase_label']}** ({det_b['g_max_fase']:.3f} G)")

    # ---- Positioning & Cycle Time ----
    st.markdown("---")
    st.subheader("🎯 Positioning & Cycle Time (Profile A)")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Conveyor Overrun", f"{overrun_a:.2f} mm")
    col6.metric("Overrun Time", f"{overrun_time_a*1000:.0f} ms")
    col7.metric("Final Part Position", f"{(pos_a[-1] + d_desliza_a):.2f} mm")
    col8.metric("Total Motion Time", f"{t_a[-1]:.2f} s")

    # ---- Comparación A vs B ----
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
    # MESA GIRATORIA (TURNING TABLE)
    # ==========================================
    st.markdown("---")
    st.header("🔄 Mesa Giratoria (Turning Table)")
    st.caption(
        f"Pieza centrada en el eje de giro. R_max (deslizamiento) = Longitud/2 = {r_max_mesa:.1f} mm. "
        f"⚠️ Distinto del Radio Riel Sensores ({st.session_state.radio_sensor_mesa:.2f} mm), que solo se usa "
        f"para convertir posiciones de sensores entre grados y mm. Ambos sensores (Reducción y Paro) están "
        f"ubicados al final del recorrido angular, separados entre sí por la distancia configurada."
    )

    if det_m['mechanically_capped']:
        st.info(
            f"ℹ️ **Mesa:** `RAMP_STOP MESA` configurado ({st.session_state.ramp_stop_mesa:.1f} °/s²) excede el "
            f"máximo mecánico alcanzable ({det_m['alpha_mechanical_max']:.1f} °/s², limitado por el piso de "
            f"{T_MIN_MECANICO_MS:.0f} ms). Se aplicó el límite mecánico: deceleración angular real = "
            f"**{det_m['alpha_stop']:.1f} °/s²** (≈{det_m['ramp_stop_real_ms']:.1f} ms)."
        )

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
            f"pero la `Distancia Sensor Reducción MESA` ({st.session_state.sensor_distance_mm_mesa:.1f} mm ≈ "
            f"{sensor_angle_mesa_actual:.1f}°) es insuficiente."
        )

    # ---- DISTANCIA ENTRE SENSORES EN MM ----
    st.markdown("---")
    st.subheader("📍 Distancia Entre Sensores (mm) — Configuración de Campo")
    st.caption(
        "Ambos sensores (Reducción y Paro) están ubicados cerca del final del recorrido angular total, "
        "montados sobre un riel fijo. Ingresa su separación física tal como se mide con cinta métrica "
        "(barra lateral) — la app la convierte automáticamente al ángulo equivalente."
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

    sd1, sd2, sd3 = st.columns(3)
    sd1.metric("Distancia Sensor Reducción MESA (entrada)", f"{st.session_state.sensor_distance_mm_mesa:.1f} mm")
    sd2.metric("→ Ángulo Equivalente Usado por el Motor", f"{sensor_angle_mesa_actual:.3f}°")
    sd3.metric("Radio Riel Sensores MESA", f"{R_sensor:.2f} mm")

    st.latex(L(
        f"~theta_{{~text{{sensor}}}} = ~text{{mm~_to~_angle}}({st.session_state.sensor_distance_mm_mesa:.1f}~,~text{{mm}}, "
        f"R={R_sensor:.2f}~,~text{{mm}}) = ~mathbf{{{sensor_angle_mesa_actual:.3f}°}}"
    ))

    # ---- EQUIVALENCIA LINEAL EN EL PUNTO CRÍTICO (R_max) ----
    st.markdown("---")
    st.subheader("📐 Equivalencia Lineal en el Punto Crítico (R_max) — Para Comparar con el Conveyor")
    st.caption(
        f"Convierte cada parámetro angular de la mesa a su equivalente lineal (mm/s, mm/s²) en el punto más "
        f"lejano de la pieza (R_max = {r_max_mesa:.1f} mm), para comparar directamente contra las mismas "
        f"unidades del conveyor lineal (Perfil A)."
    )

    eqL1, eqL2 = st.columns(2)
    with eqL1:
        st.markdown("**🔄 Mesa Giratoria (convertido a R_max)**")
        st.metric("SPEED_AUTO_FAST MESA → mm/s", f"{det_m['v_fast_lineal']:.1f} mm/s")
        st.metric("SPEED_AUTO_SLOW MESA → mm/s", f"{det_m['v_slow_lineal']:.1f} mm/s")
        st.metric("RAMP_ACCEL MESA → mm/s²", f"{det_m['a_accel_lineal']:.1f} mm/s²")
        st.metric("RAMP_DECEL MESA → mm/s²", f"{det_m['a_decel_lineal']:.1f} mm/s²")
    with eqL2:
        st.markdown("**🔵 Conveyor Lineal (Perfil A, referencia directa)**")
        st.metric("SPEED_AUTO_FAST A", f"{st.session_state.speed_fast_a:.1f} mm/s",
                   delta=f"{det_m['v_fast_lineal'] - st.session_state.speed_fast_a:+.1f} mm/s", delta_color="off")
        st.metric("SPEED_AUTO_SLOW A", f"{st.session_state.speed_slow_a:.1f} mm/s",
                   delta=f"{det_m['v_slow_lineal'] - st.session_state.speed_slow_a:+.1f} mm/s", delta_color="off")
        st.metric("RAMP_ACCEL A", f"{st.session_state.accel_a:.1f} mm/s²",
                   delta=f"{det_m['a_accel_lineal'] - st.session_state.accel_a:+.1f} mm/s²", delta_color="off")
        st.metric("RAMP_DECEL A", f"{st.session_state.decel_a:.1f} mm/s²",
                   delta=f"{det_m['a_decel_lineal'] - st.session_state.decel_a:+.1f} mm/s²", delta_color="off")

    # ---- ACELERACIÓN RESULTANTE ----
    st.markdown("---")
    st.subheader("⚠️ Aceleración Resultante en los Instantes Críticos (Tangencial + Centrípeta)")
    st.caption(
        "En los extremos entre fases, la pieza siente AMBAS componentes simultáneamente. El criterio de "
        "'SLIPPING/STABLE' mostrado abajo sigue usando solo la componente tangencial en el frenado (para no "
        "alterar el comportamiento ya validado) — este análisis adicional muestra el peor caso combinado real."
    )

    res_cols = st.columns(4)
    res_items = list(det_m['resultantes'].items())
    for col, (label, g_val) in zip(res_cols, res_items):
        is_worst = (label == det_m['peor_caso_label'])
        col.metric(label.split(" (")[0], f"{g_val:.3f} G",
                   delta="⚠️ PEOR CASO" if is_worst else None, delta_color="off")

    fs_res = det_m['factor_seguridad_resultante']
    fs_res_display = "∞" if fs_res == float('inf') else f"{fs_res:.2f}x"
    rc1, rc2 = st.columns(2)
    rc1.metric("🔺 Peor Caso Combinado (Resultante Máxima)", f"{det_m['g_res_max']:.3f} G",
               help=f"Ocurre en: {det_m['peor_caso_label']}")
    rc2.metric("🛡️ Factor de Seguridad Resultante (μ / g_res_max)", fs_res_display)

    if det_m['se_desliza_resultante'] and not desliza_m:
        st.warning(
            f"⚠️ **Diferencia detectada:** el criterio actual (solo tangencial) marca **STABLE**, pero el "
            f"criterio resultante indica que en **{det_m['peor_caso_label']}** la fuerza total "
            f"({det_m['g_res_max']:.3f} G) **supera** μ ({st.session_state.mu_mesa:.2f} G)."
        )
    elif det_m['se_desliza_resultante']:
        st.error(f"🔴 El criterio resultante confirma deslizamiento — peor caso en **{det_m['peor_caso_label']}**.")
    else:
        st.success("🟢 Incluso considerando el peor caso combinado, la pieza permanece estable en todo el ciclo.")

    # ---- OVERRUN ANGULAR Y CONVERSIÓN AL RIEL DE SENSORES ----
    st.markdown("---")
    st.subheader("📏 Overrun Angular y su Conversión al Riel de Sensores (mm)")
    st.caption(
        "Convierte el overrun angular (causado por RAMP_STOP) a la distancia lineal equivalente sobre el "
        "riel de sensores, para que mantenimiento pueda reposicionar el sensor de paro usando cinta métrica."
    )

    overrun_mm_riel = angle_to_mm(ang_overrun_m, R_sensor, modo_key)

    ov1, ov2, ov3 = st.columns(3)
    ov1.metric("Overrun Angular", f"{ang_overrun_m:.4f}°")
    ov2.metric(f"Overrun en el Riel (R={R_sensor:.2f}mm)", f"{overrun_mm_riel:.3f} mm")
    ov3.metric("Overrun Time", f"{overrun_time_m*1000:.0f} ms")

    st.markdown("**🔧 Recomendación de compensación (para preservar la posición de paro original):**")
    trigger_angle_original = st.session_state.angle_total_mesa
    trigger_angle_compensado = trigger_angle_original - ang_overrun_m
    trigger_mm_original = angle_to_mm(trigger_angle_original, R_sensor, modo_key)
    trigger_mm_compensado = angle_to_mm(trigger_angle_compensado, R_sensor, modo_key)

    comp1, comp2 = st.columns(2)
    comp1.metric("Posición Actual del Sensor de Paro", f"{trigger_mm_original:.2f} mm")
    comp2.metric("Posición Compensada Recomendada", f"{trigger_mm_compensado:.2f} mm",
                 delta=f"{(trigger_mm_compensado - trigger_mm_original):.3f} mm (mover hacia atrás)",
                 delta_color="inverse")

    st.info(
        f"💡 Para que la mesa termine deteniéndose exactamente en el mismo punto que tendría con "
        f"`RAMP_STOP MESA = 0`, mueve el **sensor de paro** físicamente **{abs(trigger_mm_compensado - trigger_mm_original):.3f} mm "
        f"hacia atrás** sobre el riel."
    )

    # ---- ANÁLISIS DE DESLIZAMIENTO (TANGENCIAL) ----
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
        "No tiene equivalente en el conveyor lineal."
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("G Centrípeta @ Velocidad Rápida", f"{det_m['g_cent_fast']:.3f} G")
    mc2.metric("Estabilidad @ Rápida", "🔴 RIESGO" if det_m['se_desliza_cruise_fast'] else "🟢 OK")
    mc3.metric("G Centrípeta @ Velocidad Lenta", f"{det_m['g_cent_slow']:.3f} G")
    mc4.metric("Estabilidad @ Lenta", "🔴 RIESGO" if det_m['se_desliza_cruise_slow'] else "🟢 OK")

    if det_m['se_desliza_cruise_fast'] or det_m['se_desliza_cruise_slow']:
        st.error(
            "🔴 **Riesgo de deslizamiento en crucero detectado.** La fuerza centrípeta a esta velocidad angular "
            "y radio supera la fricción disponible — la pieza podría deslizarse hacia afuera incluso sin frenar."
        )
    else:
        st.success("🟢 La fuerza centrípeta en ambas velocidades de crucero está dentro del límite de fricción.")

    with st.expander("🔍 Ver Desglose de Cálculo — Mesa Giratoria"):
        st.markdown("### 🧮 Memoria de Cálculo (Valores Simulados)")

        st.markdown("**Paso 0: Distancia Sensor → Ángulo**")
        st.latex(L(
            f"~theta_{{~text{{sensor}}}} = {sensor_angle_mesa_actual:.3f}° ~quad ~text{{(desde "
            f"{st.session_state.sensor_distance_mm_mesa:.1f}~,~text{{mm}} sobre riel R={R_sensor:.2f}~,~text{{mm}})}}"
        ))

        st.markdown("**Paso 1: Deceleración Angular Mecánica Máxima (Piso)**")
        st.latex(L(
            f"~alpha_{{~text{{mech,max}}}} = ~frac{{~omega_{{~text{{slow}}}}}}{{20~,~text{{ms}}}} = "
            f"~frac{{{st.session_state.speed_slow_mesa:.1f}~,°/s}}{{0.020~,~text{{s}}}} = "
            f"~mathbf{{{det_m['alpha_mechanical_max']:.1f}~,°/s^2}}"
        ))

        st.markdown("**Paso 2: Deceleración Angular Efectiva (RAMP_STOP MESA limitado por el piso mecánico)**")
        st.latex(L(
            f"~alpha_{{~text{{stop}}}} = ~min({st.session_state.ramp_stop_mesa:.1f}~,°/s^2, "
            f"{det_m['alpha_mechanical_max']:.1f}~,°/s^2) = ~mathbf{{{det_m['alpha_stop']:.1f}~,°/s^2}} "
            f"~quad (t~approx{det_m['ramp_stop_real_ms']:.1f}~,~text{{ms}})"
        ))
        st.latex(L(
            f"a_{{~text{{tan~_stop}}}} = ~alpha_{{~text{{stop}}}} ~cdot R_{{~text{{max}}}} "
            f"= {det_m['a_tan_stop']:.2f}~,~text{{mm/s}}^2"
        ))
        st.latex(L(f"g_{{~text{{conv}}}} = ~frac{{{det_m['a_tan_stop']:.2f}}}{{9810}} = ~mathbf{{{g_conv_m:.3f}~,G}}"))

        st.markdown("**Paso 3: Componente Centrípeta en Crucero**")
        st.latex(L(r"a_{~text{cent}} = ~omega^2 ~cdot R_{~text{max}} ~quad ~text{(en radianes)}"))
        st.latex(L(
            f"g_{{~text{{cent,fast}}}} = ~mathbf{{{det_m['g_cent_fast']:.3f}~,G}} ~quad "
            f"g_{{~text{{cent,slow}}}} = ~mathbf{{{det_m['g_cent_slow']:.3f}~,G}}"
        ))

        st.markdown("**Paso 4: Criterio de Deslizamiento al Frenar (tangencial simple)**")
        if desliza_m:
            st.error(f"🔴 **SLIP AL FRENAR:** g_conv ({g_conv_m:.3f} G) > μ ({st.session_state.mu_mesa:.2f} G)")
        else:
            st.success(f"🟢 **ESTABLE AL FRENAR:** g_conv ({g_conv_m:.3f} G) ≤ μ ({st.session_state.mu_mesa:.2f} G)")

    # ---- Posicionamiento y Tiempo de Ciclo — Mesa ----
    st.markdown("---")
    st.subheader("🎯 Posicionamiento y Tiempo de Ciclo — Mesa")
    mp1, mp2, mp3, mp4 = st.columns(4)
    mp1.metric("Overrun Angular", f"{ang_overrun_m:.3f}°")
    mp2.metric("Overrun Time", f"{overrun_time_m*1000:.0f} ms")
    mp3.metric("Posición Final Pieza", f"{(theta_m[-1] + det_m['deslizamiento_deg_stop']):.3f}°")
    mp4.metric("Tiempo Total de Ciclo", f"{t_m[-1]:.2f} s")


# ==========================================
# TAB 2: MATHEMATICAL & PHYSICAL BACKGROUND
# ==========================================
with tab_math:
    st.header("📐 Physics Engine & Kinematic Formulas")
    st.markdown(
        "This section details the analytical models used to simulate conveyor dynamics, "
        "deceleration forces, friction boundaries, and inertial displacement of transported parts."
    )

    st.subheader("1. Mechanical Elasticity Floor (Maximum Achievable Deceleration)")
    st.markdown(L(
        "`RAMP_STOP` is a **directly commanded deceleration** (mm/s² or °/s²), consistent with "
        "`RAMP_ACCEL`/`RAMP_DECEL` — it is NOT a time value. However, mechanical compliance (chain slack, "
        "belt stretch, chassis flex) imposes a minimum stopping time of $T_{~text{min}} = 20~,~text{ms}$, "
        "which translates into a **maximum physically achievable deceleration**:"
    ))
    st.latex(L(r"a_{~text{mech,max}} = ~frac{v_{~text{slow}}}{T_{~text{min}}}"))

    st.subheader("2. Effective Stop Deceleration")
    st.markdown(L(
        "The real applied deceleration is the commanded `RAMP_STOP`, capped by the mechanical maximum. "
        "If `RAMP_STOP` is left at 0 (no explicit PLC limit), the system uses the fastest mechanically "
        "possible stop:"
    ))
    st.latex(L(
        r"a_{~text{stop}} = ~begin{cases} a_{~text{mech,max}} & ~text{RAMP~_STOP} = 0 \\\\ "
        r"~min(~text{RAMP~_STOP}, a_{~text{mech,max}}) & ~text{RAMP~_STOP} > 0 ~end{cases}"
    ))
    st.latex(L(r"g_{~text{conv}} = ~frac{a_{~text{stop}}}{9810}"))

    st.subheader("3. Static Friction Threshold & Slip Determination")
    st.latex(L(r"a_{~text{max~_piece}} = ~mu ~cdot g = ~mu ~cdot 9810~,~text{mm/s}^2"))

    st.subheader("4. Relative Part Slip Estimation")
    st.latex(L(r"d_{~text{piece}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{max~_piece}}}"))
    st.latex(L(r"d_{~text{conveyor}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{stop}}}"))
    st.latex(L(r"~Delta d = d_{~text{piece}} - d_{~text{conveyor}}"))

    st.subheader("5. Safety Factor & Minimum Required Friction")
    st.latex(L(r"~text{Factor de Seguridad} = ~frac{~mu}{g_{~text{conv}}}"))

    st.subheader("6. Sensor Layout: Both Sensors Near End of Travel")
    st.markdown(L(
        "Both the reduction sensor and the stop sensor are positioned near the **end** of the total travel "
        "(linear or angular), separated from each other by the configured sensor distance:"
    ))
    st.latex(L(r"P_{~text{stop}} = L_{~text{total}} ~quad P_{~text{reduction}} = L_{~text{total}} - S_{~text{distance}}"))
    st.latex(L(r"~theta_{~text{stop}} = ~theta_{~text{total}} ~quad ~theta_{~text{reduction}} = ~theta_{~text{total}} - ~theta_{~text{sensor~_dist}}"))

    st.subheader("7. Sensor Distance Validation")
    st.latex(L(r"d_{~text{needed}} = ~frac{v_{~text{fast}}^2 - v_{~text{slow}}^2}{2 ~cdot ~text{RAMP~_DECEL}}"))

    st.subheader("8. Linear Conveyor: Acceleration per State Transition")
    st.markdown(L(
        "Unlike the turning table, the linear conveyor has no centripetal component — each velocity "
        "transition produces purely tangential (linear) acceleration, evaluated independently against $~mu$:"
    ))
    st.latex(L(r"g_{~text{accel}} = ~frac{~text{RAMP~_ACCEL}}{9810} ~quad ~text{(ACCEL~_FAST, arranque)}"))
    st.latex(L(r"g_{~text{decel}} = ~frac{~text{RAMP~_DECEL}}{9810} ~quad ~text{(DECEL~_TO~_SLOW, frenado a creep)}"))
    st.latex(L(r"g_{~text{stop}} = ~frac{a_{~text{stop}}}{9810} ~quad ~text{(DECEL~_TO~_STOP, frenado final)}"))
    st.latex(L(r"g_{~text{max~_fase}} = ~max~left(g_{~text{accel}}, g_{~text{decel}}, g_{~text{stop}}~right)"))

    st.subheader("9. Turning Table: Angular-to-Linear Analogy")
    st.latex(L(r"R_{~text{max}} = ~frac{~text{Longitud Total de Pieza}}{2}"))
    st.latex(L(r"v_{~text{lineal}} = ~omega_{~text{rad}} ~cdot R_{~text{max}} ~quad a_{~text{lineal}} = ~alpha_{~text{rad}} ~cdot R_{~text{max}}"))

    st.subheader("10. Turning Table: Tangential Component (Braking)")
    st.markdown(L("Same mechanical-floor logic as the linear conveyor, applied in the angular domain:"))
    st.latex(L(r"~alpha_{~text{mech,max}} = ~frac{~omega_{~text{slow}}}{T_{~text{min}}} ~quad [°/s^2]"))
    st.latex(L(
        r"~alpha_{~text{stop}} = ~begin{cases} ~alpha_{~text{mech,max}} & ~text{RAMP~_STOP~_MESA} = 0 \\\\ "
        r"~min(~text{RAMP~_STOP~_MESA}, ~alpha_{~text{mech,max}}) & ~text{RAMP~_STOP~_MESA} > 0 ~end{cases}"
    ))
    st.latex(L(r"a_{~text{tan~_stop}} = ~alpha_{~text{stop,rad}} ~cdot R_{~text{max}} ~quad [~text{mm/s}^2]"))
    st.latex(L(r"g_{~text{conv}} = ~frac{a_{~text{tan~_stop}}}{9810}"))

    st.subheader("11. Turning Table: Centripetal Component (Constant Speed Cruise)")
    st.markdown(L(
        "**This has no equivalent in the linear conveyor model.** Even at perfectly constant angular velocity, "
        "any point at radius $R$ experiences a centripetal acceleration directed toward the rotation center:"
    ))
    st.latex(L(r"a_{~text{cent}} = ~omega_{~text{rad}}^2 ~cdot R_{~text{max}} ~quad [~text{mm/s}^2]"))
    st.latex(L(r"g_{~text{cent}} = ~frac{a_{~text{cent}}}{9810}"))

    st.subheader("12. Turning Table: Combined Resultant Acceleration at Critical Instants")
    st.latex(L(r"a_{~text{resultante}} = ~sqrt{a_{~text{tangencial}}^2 + a_{~text{centrípeta}}^2}"))

    st.subheader("13. Scope Limitation: Free-Sliding Dynamics on a Rotating Frame")
    st.markdown(L(
        "A fully rigorous simulation of an object sliding freely on a rotating platform (once friction is "
        "exceeded) requires solving motion in a non-inertial rotating reference frame, which introduces "
        "Coriolis and Euler pseudo-forces. This simulator intentionally simplifies that scenario."
    ))

    st.subheader("14. Sensor Rail: Angular-to-Linear Conversion for Maintenance")
    st.latex(L(r"~theta_{~text{sensor}} = ~frac{s_{~text{mm}}}{R_{~text{sensor}}} ~quad ~text{(Arco, en radianes, luego convertido a grados)}"))
    st.latex(L(r"~theta_{~text{sensor}} = 2 ~arcsin~left(~frac{s_{~text{mm}}}{2R_{~text{sensor}}}~right) ~quad ~text{(Cuerda)}"))
    st.latex(L(r"s_{~text{arc}} = R_{~text{sensor}} ~cdot ~theta_{~text{rad}}"))
    st.latex(L(r"s_{~text{chord}} = 2R_{~text{sensor}} ~sin~left(~frac{~theta_{~text{rad}}}{2}~right)"))
