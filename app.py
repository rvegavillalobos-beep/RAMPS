import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# BACKSLASH-SAFE LATEX HELPER
# ==========================================
BS = chr(92)


def L(template: str) -> str:
    """Replaces '~' with a real backslash to build valid LaTeX."""
    return template.replace("~", BS)


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Kinematic Simulator — Turntable & Conveyor",
    page_icon="⚙️",
    layout="wide",
)

# ==========================================
# ANGLE <-> MM CONVERSION ON A FIXED-RADIUS RAIL
# ==========================================
def angle_to_mm(angle_deg, radius, mode="Arc (R·θ)"):
    theta_rad = np.radians(angle_deg)
    if mode.startswith("Arc"):
        return radius * theta_rad
    else:
        return 2.0 * radius * np.sin(theta_rad / 2.0)


def mm_to_angle(mm, radius, mode="Arc (R·θ)"):
    if radius <= 0:
        return 0.0
    if mode.startswith("Arc"):
        theta_rad = mm / radius
    else:
        ratio = np.clip(mm / (2.0 * radius), -1.0, 1.0)
        theta_rad = 2.0 * np.arcsin(ratio)
    return np.degrees(theta_rad)


# ==========================================
# CONSTANTS
# ==========================================
T_MIN_MECANICO_MS = 20.0
T_MIN_MECANICO_S = T_MIN_MECANICO_MS / 1000.0

# Colors (consistent across the app)
COL_A = "#1f77b4"
COL_B = "#9467bd"
COL_SENSOR_RED = "#ff7f0e"
COL_SENSOR_STOP = "#d62728"
COL_OK = "#2ca02c"
COL_SLIP = "#d62728"
COL_TAN = "#ff7f0e"
COL_CENT = "#17becf"
COL_NEUTRO = "#7f7f7f"

# PLC state machine states (same order in both engines)
FASES = ["ACCEL_FAST", "CRUISE_FAST", "DECEL_TO_SLOW", "CRUISE_SLOW", "DECEL_TO_STOP", "DONE"]
FASE_CODE = {f: i for i, f in enumerate(FASES)}
FASE_NOMBRE = {
    "ACCEL_FAST": "Start-up",
    "CRUISE_FAST": "Fast cruise",
    "DECEL_TO_SLOW": "Braking to creep",
    "CRUISE_SLOW": "Slow cruise (creep)",
    "DECEL_TO_STOP": "Final stop",
    "DONE": "Stopped",
}
FASE_CORTO = {
    "ACCEL_FAST": "ACCEL",
    "CRUISE_FAST": "CRUISE",
    "DECEL_TO_SLOW": "DECEL",
    "CRUISE_SLOW": "CREEP",
    "DECEL_TO_STOP": "STOP",
}

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
    "modo_conversion_mesa": "Arc (R·θ) — recommended",
}
MODOS_CONVERSION = ["Arc (R·θ) — recommended", "Chord (2R·sin(θ/2))"]
# Re-assigning every value on each run keeps the parameters when switching sections
# (Streamlit drops the state of widgets that are not drawn on the active page).
for key, val in defaults.items():
    st.session_state[key] = st.session_state.get(key, val)
# Sessions started on the Spanish version may hold the old option text
if st.session_state.modo_conversion_mesa not in MODOS_CONVERSION:
    st.session_state.modo_conversion_mesa = MODOS_CONVERSION[0 if str(st.session_state.modo_conversion_mesa).startswith("Arc") else 1]
if "_comparar_prev" not in st.session_state:
    st.session_state._comparar_prev = False


# ==========================================
# REALISTIC KINEMATIC CALCULATION ENGINE (LINEAR CONVEYOR)
# ==========================================
@st.cache_data(show_spinner=False)
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist, ramp_stop_accel, mu):
    dt = 0.001
    t_max = 30.0
    steps = int(t_max / dt)
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)
    # Per-step log: state that governed the step and commanded acceleration (mm/s²)
    fase = np.zeros(steps, dtype=np.int8)
    a_cmd = np.zeros(steps)
    a_cmd[0] = accel

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
        "ACCEL_FAST (start-up)": g_accel,
        "DECEL_TO_SLOW (braking to creep)": g_decel,
        "DECEL_TO_STOP (final stop)": g_conveyor,
    }
    peor_fase_label = max(fases_g, key=fases_g.get)
    g_max_fase = fases_g[peor_fase_label]
    se_desliza_alguna_fase = g_max_fase > mu
    factor_seguridad_fase = (mu / g_max_fase) if g_max_fase > 0 else float('inf')

    a_por_estado = {
        "ACCEL_FAST": accel, "CRUISE_FAST": 0.0, "DECEL_TO_SLOW": -decel,
        "CRUISE_SLOW": 0.0, "DECEL_TO_STOP": -a_stop_conveyor, "DONE": 0.0,
    }

    for i in range(1, steps):
        t[i] = t[i - 1] + dt

        if state not in ("DECEL_TO_STOP", "DONE") and p >= pos_stop:
            t_sensor_stop = t[i]
            state = "DECEL_TO_STOP"

        estado_paso = state

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
        fase[i] = FASE_CODE[estado_paso]
        a_cmd[i] = a_por_estado[estado_paso]

        if state == "DONE" and i > 50 and np.all(vel[i - 20:i] == 0):
            t = t[:i + 1]
            pos = pos[:i + 1]
            vel = vel[:i + 1]
            fase = fase[:i + 1]
            a_cmd = a_cmd[:i + 1]
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
        "fase": fase,
        "a_cmd": a_cmd,
    }

    return (t, pos, vel, t_sensor_red, t_sensor_stop, dist_overrun_conveyor, overrun_time,
            g_conveyor, g_pieza_real, se_desliza, deslizamiento_mm, details)


# ==========================================
# ANGULAR KINEMATIC CALCULATION ENGINE (TURNTABLE)
# ==========================================
@st.cache_data(show_spinner=False)
def calcular_perfil_mesa(omega_fast, omega_slow, alpha_accel, alpha_decel,
                          angle_total, angle_sensor_dist, ramp_stop_alpha, mu, r_max):
    dt = 0.001
    t_max = 30.0
    steps = int(t_max / dt)
    t = np.zeros(steps)
    theta = np.zeros(steps)
    omega = np.zeros(steps)
    # Per-step log: state that governed the step and commanded angular acceleration (°/s²)
    fase = np.zeros(steps, dtype=np.int8)
    alpha_cmd = np.zeros(steps)
    alpha_cmd[0] = alpha_accel

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
        "End of ACCEL_FAST (ω=ω_fast, tan=accel)": g_res_fin_accel,
        "Start of DECEL_TO_SLOW (ω=ω_fast, tan=decel)": g_res_ini_decel,
        "End of DECEL_TO_SLOW (ω=ω_slow, tan=decel)": g_res_fin_decel,
        "Start of DECEL_TO_STOP (ω=ω_slow, tan=stop)": g_res_ini_stop,
    }
    # Components (tangential, centripetal) in G at each critical instant — for plotting
    componentes = {
        "End of ACCEL_FAST (ω=ω_fast, tan=accel)": (a_accel_lineal / 9810.0, g_cent_fast),
        "Start of DECEL_TO_SLOW (ω=ω_fast, tan=decel)": (a_decel_lineal / 9810.0, g_cent_fast),
        "End of DECEL_TO_SLOW (ω=ω_slow, tan=decel)": (a_decel_lineal / 9810.0, g_cent_slow),
        "Start of DECEL_TO_STOP (ω=ω_slow, tan=stop)": (g_conv_mesa, g_cent_slow),
    }
    peor_caso_label = max(resultantes, key=resultantes.get)
    g_res_max = resultantes[peor_caso_label]
    se_desliza_resultante = g_res_max > mu

    if omega_fast > omega_slow and alpha_decel > 0:
        angle_needed_decel = (omega_fast ** 2 - omega_slow ** 2) / (2.0 * alpha_decel)
    else:
        angle_needed_decel = 0.0
    insufficient_angle = angle_needed_decel > angle_sensor_dist

    alpha_por_estado = {
        "ACCEL_FAST": alpha_accel, "CRUISE_FAST": 0.0, "DECEL_TO_SLOW": -alpha_decel,
        "CRUISE_SLOW": 0.0, "DECEL_TO_STOP": -alpha_stop, "DONE": 0.0,
    }

    for i in range(1, steps):
        t[i] = t[i - 1] + dt

        if state not in ("DECEL_TO_STOP", "DONE") and th >= angle_stop:
            t_sensor_stop = t[i]
            state = "DECEL_TO_STOP"

        estado_paso = state

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
        fase[i] = FASE_CODE[estado_paso]
        alpha_cmd[i] = alpha_por_estado[estado_paso]

        if state == "DONE" and i > 50 and np.all(omega[i - 20:i] == 0):
            t = t[:i + 1]
            theta = theta[:i + 1]
            omega = omega[:i + 1]
            fase = fase[:i + 1]
            alpha_cmd = alpha_cmd[:i + 1]
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
        "componentes": componentes,
        "peor_caso_label": peor_caso_label,
        "g_res_max": g_res_max,
        "se_desliza_resultante": se_desliza_resultante,
        "factor_seguridad_resultante": factor_seguridad_resultante,
        "g_res_fin_accel": g_res_fin_accel,
        "g_res_ini_decel": g_res_ini_decel,
        "g_res_fin_decel": g_res_fin_decel,
        "g_res_ini_stop": g_res_ini_stop,
        "fase": fase,
        "alpha_cmd": alpha_cmd,
    }

    return (t, theta, omega, t_sensor_red, t_sensor_stop, angle_overrun, overrun_time,
            g_conv_mesa, se_desliza_stop, deslizamiento_mm_stop, details)



# ==========================================
# PRESENTATION HELPERS
# ==========================================
def fmt_fs(fs):
    return "∞" if fs == float('inf') else f"{fs:.2f}x"


def estado_txt(desliza):
    return "🔴 SLIPPING" if desliza else "🟢 STABLE"


def indices_grafica(fase, paso=5):
    """Downsample for plotting without losing phase changes (curves are piecewise linear)."""
    n = len(fase)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[::paso] = True
    keep[-1] = True
    cambios = np.flatnonzero(np.diff(fase))
    keep[cambios] = True
    keep[cambios + 1] = True
    return np.flatnonzero(keep)


def segmentos_fase(t, fase):
    """Contiguous stretches of each PLC state: start/end index and times."""
    segs = []
    n = len(fase)
    if n == 0:
        return segs
    cambios = np.flatnonzero(np.diff(fase)) + 1
    inicios = np.r_[0, cambios]
    finales = np.r_[cambios, n]
    for s, e in zip(inicios, finales):
        nombre = FASES[int(fase[s])]
        if nombre == "DONE":
            continue
        i0 = max(int(s) - 1, 0)  # instant the phase begins
        i1 = int(e) - 1          # last step governed by the phase
        segs.append({"fase": nombre, "i0": i0, "i1": i1, "t0": float(t[i0]), "t1": float(t[i1])})
    return segs


def bandas_fase(fig, segs, g_fase, mu, t_total, fila_etiquetas=2):
    """Shade each phase on both rows (red if it exceeds μ) and name it on the given row."""
    xref = "x" if fila_etiquetas == 1 else f"x{fila_etiquetas}"
    yref = "y domain" if fila_etiquetas == 1 else f"y{fila_etiquetas} domain"
    for sg in segs:
        g = g_fase(sg)
        desliza = g > mu
        if desliza:
            fig.add_vrect(x0=sg["t0"], x1=sg["t1"], fillcolor=COL_SLIP, opacity=0.12,
                          line_width=0, layer="below", row="all", col=1)
        elif sg["fase"] not in ("CRUISE_FAST", "CRUISE_SLOW"):
            fig.add_vrect(x0=sg["t0"], x1=sg["t1"], fillcolor=COL_NEUTRO, opacity=0.07,
                          line_width=0, layer="below", row="all", col=1)
        if t_total > 0 and (sg["t1"] - sg["t0"]) / t_total >= 0.07:
            fig.add_annotation(
                x=(sg["t0"] + sg["t1"]) / 2, y=0.98, xref=xref, yref=yref,
                text=FASE_CORTO[sg["fase"]], showarrow=False, yanchor="top",
                font=dict(size=10, color=COL_SLIP if desliza else COL_NEUTRO),
            )


COL_OK_TXT = "#1e7b34"     # darker green for better text contrast
COL_BORDE_OK = "#9ca3af"


def ticks_bonitos(lo, hi, max_ticks=5):
    """'Round' ticks only over the data range (the label strip stays free of gridlines)."""
    span = hi - lo
    if span <= 0:
        return None
    mag = 10 ** np.floor(np.log10(span / max_ticks))
    for mult in (1, 2, 2.5, 5, 10):
        paso = mult * mag
        if span / paso <= max_ticks:
            break
    inicio = np.ceil(lo / paso - 1e-9) * paso
    return [round(float(v), 6) for v in np.arange(inicio, hi + paso * 1e-6, paso)]


def distribuir_x(deseadas, lo, hi, sep):
    """Spread labels along one row, keeping their order and a minimum spacing between centers."""
    n = len(deseadas)
    if n == 0:
        return []
    orden = np.argsort(deseadas, kind="stable")
    x = np.clip(np.asarray(deseadas, dtype=float)[orden], lo, hi)
    if n > 1:
        sep = min(sep, (hi - lo) / (n - 1))
        for i in range(1, n):
            x[i] = max(x[i], x[i - 1] + sep)
        if x[-1] > hi:
            x[-1] = hi
            for i in range(n - 2, -1, -1):
                x[i] = min(x[i], x[i + 1] - sep)
    out = np.empty(n)
    out[orden] = x
    return out.tolist()


def texto_etiqueta(titulo, valor, desliza, detalle, color_titulo="#374151"):
    color_val = COL_SLIP if desliza else COL_OK_TXT
    return (f"<span style='color:{color_titulo}'><b>{titulo}</b></span><br>"
            f"<span style='font-size:14px;color:{color_val}'><b>{valor}</b></span><br>"
            f"<span style='font-size:10px;color:#6b7280'>{detalle}</span>")


def fila_etiquetas(fig, items, y_fila, t_total, row=2):
    """Draw one row of labels at height y_fila, each with an arrow to the point of its phase."""
    if not items or t_total <= 0:
        return
    xs = distribuir_x([it["x"] for it in items], 0.07 * t_total, 0.93 * t_total, 0.17 * t_total)
    xr = "x" if row == 1 else f"x{row}"
    yr = "y" if row == 1 else f"y{row}"
    for it, x_lab in zip(items, xs):
        borde = COL_SLIP if it["desliza"] else COL_BORDE_OK
        fig.add_annotation(
            x=it["x"], y=it["y"], xref=xr, yref=yr,
            ax=x_lab, ay=y_fila, axref=xr, ayref=yr,
            text=it["texto"], align="center", showarrow=True,
            arrowhead=2, arrowsize=0.9, arrowwidth=1.2, arrowcolor=borde, standoff=3,
            bgcolor="rgba(255,255,255,0.95)", bordercolor=borde,
            borderwidth=2 if it.get("peor") else 1, borderpad=4,
            font=dict(size=11, color="#1f2937"),
        )


def componentes_fase_mesa(m, g_cent, sg):
    """(tangential G, max centripetal G, max resultant G) of a phase — same criterion for chart and table."""
    a, b = sg["i0"], sg["i1"]
    tan = abs(np.radians(m["det"]["alpha_cmd"][b])) * m["r_max"] / 9810.0
    cent = float(np.max(g_cent[a:b + 1]))
    return tan, cent, float(np.sqrt(tan ** 2 + cent ** 2))


def entrada_leyenda(fig, nombre, color, dash, **pos):
    """Legend-only entry (sensors, μ): takes no space on the plot and does not affect the axes."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nombre, hoverinfo="skip",
                             line=dict(color=color, width=2, dash=dash)), **pos)


def leyenda_abajo(fig, height, legend_y=-0.1):
    fig.update_layout(
        height=height, hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=legend_y, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(b=120, t=70),
    )


# ==========================================
# SIMULATIONS (wrappers that read the session parameters)
# ==========================================
def simular_conveyor(sufijo):
    ss = st.session_state
    (t, pos, vel, t_red, t_stop, overrun, overrun_time,
     g_conv, g_pieza, desliza, d_desliza, det) = calcular_perfil(
        ss[f"speed_fast_{sufijo}"], ss[f"speed_slow_{sufijo}"], ss[f"accel_{sufijo}"],
        ss[f"decel_{sufijo}"], ss.conveyor_length, ss[f"sensor_distance_{sufijo}"],
        ss[f"ramp_stop_{sufijo}"], ss[f"mu_{sufijo}"]
    )
    return dict(t=t, pos=pos, vel=vel, t_red=t_red, t_stop=t_stop, overrun=overrun,
                overrun_time=overrun_time, g_conv=g_conv, g_pieza=g_pieza, desliza=desliza,
                d_desliza=d_desliza, det=det, mu=ss[f"mu_{sufijo}"])


def modo_conversion_key():
    return "Arc (R·θ)" if st.session_state.modo_conversion_mesa.startswith("Arc") else "Chord (2R·sin(θ/2))"


def simular_mesa():
    ss = st.session_state
    r_max = ss.pieza_longitud_mesa / 2.0
    sensor_angle = mm_to_angle(ss.sensor_distance_mm_mesa, ss.radio_sensor_mesa, modo_conversion_key())
    (t, theta, omega, t_red, t_stop, ang_overrun, overrun_time,
     g_conv, desliza, d_desliza_mm, det) = calcular_perfil_mesa(
        ss.speed_fast_mesa, ss.speed_slow_mesa, ss.accel_mesa, ss.decel_mesa,
        ss.angle_total_mesa, sensor_angle, ss.ramp_stop_mesa, ss.mu_mesa, r_max
    )
    return dict(t=t, theta=theta, omega=omega, t_red=t_red, t_stop=t_stop,
                ang_overrun=ang_overrun, overrun_time=overrun_time, g_conv=g_conv,
                desliza=desliza, d_desliza_mm=d_desliza_mm, det=det, r_max=r_max,
                sensor_angle=sensor_angle, mu=ss.mu_mesa,
                theta_red=float(np.interp(t_red, t, theta)),
                theta_stop=float(np.interp(t_stop, t, theta)))


def series_g_mesa(m):
    """Tangential, centripetal and resultant G at R_max for every instant of the simulation."""
    r = m["r_max"]
    g_tan = np.abs(np.radians(m["det"]["alpha_cmd"])) * r / 9810.0
    g_cent = (np.radians(m["omega"]) ** 2) * r / 9810.0
    g_res = np.sqrt(g_tan ** 2 + g_cent ** 2)
    return g_tan, g_cent, g_res


# ==========================================
# PAGE: TURNTABLE
# ==========================================
def sidebar_mesa():
    sb = st.sidebar
    sb.header("🔄 Parameters — Turntable")

    sb.subheader("Kinematics (PLC)")
    sb.number_input("SPEED_AUTO_FAST TABLE (°/s)", step=5.0, key="speed_fast_mesa")
    sb.number_input("SPEED_AUTO_SLOW TABLE (°/s)", step=1.0, key="speed_slow_mesa")
    sb.number_input("RAMP_ACCEL TABLE (°/s²)", step=10.0, key="accel_mesa")
    sb.number_input("RAMP_DECEL TABLE (°/s²)", step=10.0, key="decel_mesa")
    sb.number_input(
        "RAMP_STOP TABLE (°/s²)", step=25.0, min_value=0.0, key="ramp_stop_mesa",
        help="Angular deceleration commanded directly by the PLC for the final stop — same kind of "
             "parameter as RAMP_ACCEL/RAMP_DECEL TABLE (NOT a time). A mechanical floor limits the maximum "
             "physically achievable angular deceleration to ω_slow / 20 ms. If left at 0, the mechanical "
             "maximum is assumed."
    )
    sb.number_input("Total Rotation Angle TABLE (°)", step=15.0, key="angle_total_mesa",
                    help="Angular equivalent of 'Total Conveyor Length' — open")

    sb.subheader("Sensors on the rail")
    sb.number_input("Sensor Rail Radius TABLE (mm)", step=5.0, min_value=1.0, key="radio_sensor_mesa",
                    help="Physical radius where the slowdown/stop sensors are mounted on the rail — "
                         "different from the part radius")
    sb.radio(
        "Angle ↔ mm conversion formula",
        MODOS_CONVERSION,
        key="modo_conversion_mesa",
        help="Arc = exact distance if the rail follows the curvature at that radius (typical case). "
             "Chord = straight-line distance between two points on the circle (useful if the rail is perfectly straight)."
    )
    sb.number_input(
        "Slowdown Sensor Distance TABLE (mm)", step=5.0, min_value=0.0, key="sensor_distance_mm_mesa",
        help="Physical separation, measured along the rail, between the Slowdown Sensor and the Stop Sensor. "
             "Both sensors are located near the end of the total angular travel."
    )
    _ang = mm_to_angle(st.session_state.sensor_distance_mm_mesa, st.session_state.radio_sensor_mesa,
                       modo_conversion_key())
    sb.caption(f"↳ Angular equivalent: **{_ang:.3f}°** (mode: {modo_conversion_key()})")

    sb.subheader("Part")
    sb.number_input("Total Part Length TABLE (mm)", step=10.0, min_value=1.0, key="pieza_longitud_mesa",
                    help="The part is centered on the rotation axis. R_max = Length / 2 "
                         "(used ONLY for the slip analysis)")
    sb.number_input("Friction Coefficient μ TABLE", step=0.01, min_value=0.01, max_value=1.0, key="mu_mesa")


def figura_movimiento_mesa(m):
    det = m["det"]
    t, omega, theta = m["t"], m["omega"], m["theta"]
    g_tan, g_cent, g_res = series_g_mesa(m)
    idx = indices_grafica(det["fase"])
    nombres = np.array([FASE_NOMBRE[FASES[c]] for c in det["fase"]], dtype=object)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.46, 0.54], vertical_spacing=0.08,
        subplot_titles=("Turntable angular velocity (°/s)",
                        f"Acceleration on the part at R_max = {m['r_max']:.0f} mm (G) — green zone = within μ"),
    )

    fig.add_trace(go.Scatter(
        x=t[idx], y=omega[idx], mode="lines", name="Angular velocity ω",
        line=dict(color=COL_A, width=3),
        customdata=np.column_stack([theta[idx], nombres[idx]]),
        hovertemplate="ω = %{y:.2f} °/s · θ = %{customdata[0]:.2f}° · %{customdata[1]}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t[idx], y=g_tan[idx], mode="lines", name="Tangential (ramp)",
        line=dict(color=COL_TAN, width=1.5, shape="vh"), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t[idx], y=g_cent[idx], mode="lines", name="Centripetal (ω²·R)",
        line=dict(color=COL_CENT, width=1.5), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t[idx], y=g_res[idx], mode="lines", name="Resultant",
        line=dict(color="#444444", width=2.5), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)

    mu = m["mu"]
    g_top = max(float(np.max(g_res)) if len(g_res) else 0.0, mu, 1e-3)
    fig.add_hrect(y0=0, y1=mu, fillcolor=COL_OK, opacity=0.08, line_width=0, layer="below", row=2, col=1)
    fig.add_hline(y=mu, line_width=2, line_dash="dash", line_color=COL_SLIP, row=2, col=1)
    entrada_leyenda(fig, f"Friction limit μ = {mu:.2f}", COL_SLIP, "dash", row=2, col=1)

    for fila in (1, 2):
        fig.add_vline(x=m["t_red"], line_width=2, line_dash="dot", line_color=COL_SENSOR_RED, row=fila, col=1)
        fig.add_vline(x=m["t_stop"], line_width=2, line_dash="dash", line_color=COL_SENSOR_STOP, row=fila, col=1)
    entrada_leyenda(fig, f"Slowdown Sensor · {m['t_red']:.2f}s · {m['theta_red']:.2f}°",
                    COL_SENSOR_RED, "dot", row=1, col=1)
    entrada_leyenda(fig, f"Stop Sensor · {m['t_stop']:.2f}s · {m['theta_stop']:.2f}°",
                    COL_SENSOR_STOP, "dash", row=1, col=1)

    segs = segmentos_fase(t, det["fase"])
    comps = {id(sg): componentes_fase_mesa(m, g_cent, sg) for sg in segs}
    t_total = float(t[-1])
    bandas_fase(fig, segs, lambda sg: comps[id(sg)][2], mu, t_total, fila_etiquetas=1)

    # ---- Value label per phase (resultant and its components), always above the curves ----
    peor = max(segs, key=lambda sg: comps[id(sg)][2]) if segs else None
    items = []
    for sg in segs:
        tan, cent, res = comps[id(sg)]
        a, b = sg["i0"], sg["i1"]
        j0 = a + 1 if b > a else b
        tramo = g_res[j0:b + 1]
        # Arrow to the point of maximum resultant; if the phase is constant, to its center
        j = (j0 + b) // 2 if np.ptp(tramo) < 1e-4 else j0 + int(np.argmax(tramo))
        items.append({
            "x": float(t[j]), "y": float(g_res[j]), "desliza": res > mu,
            "peor": sg is peor and res > 0,
            "texto": texto_etiqueta(FASE_CORTO[sg["fase"]], f"{res:.3f} G", res > mu,
                                    f"tan {tan:.3f} · cent {cent:.3f}"),
        })
    fila_etiquetas(fig, items, y_fila=g_top * 1.55, t_total=t_total)

    w_max = max(float(np.max(omega)) if len(omega) else 0.0, 1e-3)
    fig.update_yaxes(title_text="°/s", range=[-0.04 * w_max, 1.2 * w_max], row=1, col=1)
    fig.update_yaxes(title_text="G", range=[0, g_top * 1.95], tickvals=ticks_bonitos(0, g_top), row=2, col=1)
    fig.update_xaxes(range=[0, t_total])
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    leyenda_abajo(fig, 800)
    return fig


def tabla_fases_mesa(m):
    det = m["det"]
    _, g_cent, _ = series_g_mesa(m)
    filas = []
    for sg in segmentos_fase(m["t"], det["fase"]):
        a, b = sg["i0"], sg["i1"]
        tan, cent, res = componentes_fase_mesa(m, g_cent, sg)
        filas.append({
            "Phase": FASE_NOMBRE[sg["fase"]],
            "PLC state": sg["fase"],
            "Start (s)": round(sg["t0"], 3),
            "Duration (s)": round(sg["t1"] - sg["t0"], 3),
            "Rotation (°)": round(float(m["theta"][b] - m["theta"][a]), 3),
            "ω start → end (°/s)": f"{m['omega'][a]:.1f} → {m['omega'][b]:.1f}",
            "Tangential G": round(tan, 3),
            "Max centripetal G": round(cent, 3),
            "Max resultant G": round(res, 3),
            "Status": "🔴 Exceeds μ" if res > m["mu"] else "🟢 OK",
        })
    return filas


def figura_instantes_mesa(m):
    det = m["det"]
    mu = m["mu"]
    etiquetas = [k.split(" (")[0].replace(" of ", "<br>") for k in det["resultantes"]]
    tan = [c[0] for c in det["componentes"].values()]
    cent = [c[1] for c in det["componentes"].values()]
    res = list(det["resultantes"].values())

    fig = go.Figure()
    fig.add_trace(go.Bar(x=etiquetas, y=tan, name="Tangential", marker_color=COL_TAN, opacity=0.55,
                         hovertemplate="%{y:.3f} G"))
    fig.add_trace(go.Bar(x=etiquetas, y=cent, name="Centripetal", marker_color=COL_CENT, opacity=0.55,
                         hovertemplate="%{y:.3f} G"))
    fig.add_trace(go.Bar(x=etiquetas, y=res, name="Resultant",
                         marker_color=[COL_SLIP if g > mu else COL_OK for g in res],
                         text=[f"{g:.3f} G" for g in res], textposition="outside",
                         hovertemplate="%{y:.3f} G"))
    fig.add_hline(y=mu, line_width=2, line_dash="dash", line_color=COL_SLIP)
    entrada_leyenda(fig, f"Friction limit μ = {mu:.2f}", COL_SLIP, "dash")
    fig.update_layout(
        title="Critical instants: components and resultant vs μ", barmode="group", height=420,
        yaxis_title="G", yaxis_range=[0, max(max(res), mu) * 1.3],
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(b=90, t=70),
    )
    return fig


def figura_equivalencia(m, conv_a):
    v_mesa = np.radians(m["omega"]) * m["r_max"]
    idx_m = indices_grafica(m["det"]["fase"])
    idx_c = indices_grafica(conv_a["det"]["fase"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["t"][idx_m], y=v_mesa[idx_m], mode="lines",
                             name=f"Turntable — linear velocity at R_max ({m['r_max']:.0f} mm)",
                             line=dict(color=COL_A, width=3), hovertemplate="%{y:.1f} mm/s"))
    fig.add_trace(go.Scatter(x=conv_a["t"][idx_c], y=conv_a["vel"][idx_c], mode="lines",
                             name="Conveyor — Profile A", line=dict(color=COL_NEUTRO, width=2, dash="dash"),
                             hovertemplate="%{y:.1f} mm/s"))
    fig.update_layout(title="Linear velocity (mm/s) vs time — turntable at R_max vs conveyor",
                      xaxis_title="Time (s)", yaxis_title="mm/s")
    leyenda_abajo(fig, 440, legend_y=-0.22)
    return fig


def pagina_mesa():
    sidebar_mesa()
    ss = st.session_state
    m = simular_mesa()
    det_m = m["det"]
    t_m, theta_m = m["t"], m["theta"]
    r_max_mesa = m["r_max"]
    R_sensor = ss.radio_sensor_mesa
    modo_key = modo_conversion_key()

    st.title("🔄 Turntable")
    st.caption(
        f"Part centered on the rotation axis. R_max (slip) = Length/2 = {r_max_mesa:.1f} mm. "
        f"⚠️ Different from the Sensor Rail Radius ({R_sensor:.2f} mm), which is only used "
        f"to convert sensor positions between degrees and mm."
    )

    # ---- Configuration alerts ----
    if det_m['insufficient_angle']:
        st.warning(
            f"⚠️ **Inconsistent configuration:** `RAMP_DECEL TABLE` needs "
            f"**{det_m['angle_needed_decel']:.1f}°** to go from SPEED_AUTO_FAST down to SPEED_AUTO_SLOW, "
            f"but the `Slowdown Sensor Distance TABLE` ({ss.sensor_distance_mm_mesa:.1f} mm ≈ "
            f"{m['sensor_angle']:.1f}°) is not enough."
        )
    if det_m['mechanically_capped']:
        st.info(
            f"ℹ️ Configured `RAMP_STOP TABLE` ({ss.ramp_stop_mesa:.1f} °/s²) exceeds the "
            f"achievable mechanical maximum ({det_m['alpha_mechanical_max']:.1f} °/s², limited by the "
            f"{T_MIN_MECANICO_MS:.0f} ms floor). The mechanical limit was applied: actual angular deceleration = "
            f"**{det_m['alpha_stop']:.1f} °/s²** (≈{det_m['ramp_stop_real_ms']:.1f} ms)."
        )

    # ---- Summary ----
    with st.container(border=True):
        st.markdown("**Cycle summary**")
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Status (combined worst case)", estado_txt(det_m['se_desliza_resultante']),
                  help="Most complete criterion: tangential + centripetal at the critical instants.")
        v2.metric("Max acceleration on the part", f"{det_m['g_res_max']:.3f} G",
                  help=f"Occurs at: {det_m['peor_caso_label']}")
        v3.metric("Safety factor", fmt_fs(det_m['factor_seguridad_resultante']))
        v4.metric("Cycle time", f"{t_m[-1]:.2f} s")
        v5.metric("Angular overrun", f"{m['ang_overrun']:.3f}°")

    tab_mov, tab_slip, tab_sens, tab_eq, tab_fund = st.tabs([
        "📈 Motion", "⚠️ Slip", "📍 Sensors & adjustment", "🔁 Conveyor equivalence",
        "📚 Fundamentals",
    ])

    # ------------------------------------------
    with tab_mov:
        st.plotly_chart(figura_movimiento_mesa(m))
        st.caption(
            "Labels: maximum resultant acceleration of each phase with its tangential (ramp) and "
            "centripetal (ω²·R) components — green = within μ, red = exceeds μ, thick border = most critical phase. "
            "Bands: PLC phases. Hover over the velocity curve to see the angle θ at each instant."
        )

        st.subheader("🎯 Positioning & cycle time")
        mp1, mp2, mp3, mp4, mp5 = st.columns(5)
        mp1.metric("Angular Overrun", f"{m['ang_overrun']:.3f}°")
        mp2.metric("Overrun Time", f"{m['overrun_time']*1000:.0f} ms")
        mp3.metric("Final Table Angle", f"{theta_m[-1]:.3f}°",
                   help=f"Should be the Total Rotation Angle ({ss.angle_total_mesa:.1f}°) plus the overrun.")
        mp4.metric("Final Part Position", f"{(theta_m[-1] + det_m['deslizamiento_deg_stop']):.3f}°")
        mp5.metric("Total Cycle Time", f"{t_m[-1]:.2f} s")

        st.subheader("🧭 Cycle phases")
        st.dataframe(tabla_fases_mesa(m), hide_index=True)

    # ------------------------------------------
    with tab_slip:
        st.plotly_chart(figura_instantes_mesa(m))

        st.subheader("Resultant acceleration at the critical instants (tangential + centripetal)")
        st.caption(
            "At the boundaries between phases the part feels BOTH components at the same time. The "
            "'STABLE/SLIPPING' criterion for the final stop (below) uses only the tangential component — this "
            "analysis shows the actual combined worst case."
        )
        res_cols = st.columns(4)
        for col, (label, g_val) in zip(res_cols, det_m['resultantes'].items()):
            is_worst = (label == det_m['peor_caso_label'])
            col.metric(label.split(" (")[0], f"{g_val:.3f} G",
                       delta="⚠️ WORST CASE" if is_worst else None, delta_color="off", help=label)

        rc1, rc2 = st.columns(2)
        rc1.metric("🔺 Combined Worst Case (Max Resultant)", f"{det_m['g_res_max']:.3f} G",
                   help=f"Occurs at: {det_m['peor_caso_label']}")
        rc2.metric("🛡️ Resultant Safety Factor (μ / g_res_max)", fmt_fs(det_m['factor_seguridad_resultante']))

        if det_m['se_desliza_resultante'] and not m['desliza']:
            st.warning(
                f"⚠️ **Difference detected:** the current criterion (tangential only) says **STABLE**, but the "
                f"resultant criterion shows that at **{det_m['peor_caso_label']}** the total force "
                f"({det_m['g_res_max']:.3f} G) **exceeds** μ ({ss.mu_mesa:.2f} G)."
            )
        elif det_m['se_desliza_resultante']:
            st.error(f"🔴 The resultant criterion confirms slip — worst case at **{det_m['peor_caso_label']}**.")
        else:
            st.success("🟢 Even considering the combined worst case, the part stays stable throughout the cycle.")

        st.divider()
        st.subheader("Final stop (tangential component)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tangential Deceleration (stop)", f"{m['g_conv']:.3f} G")
        m2.metric("Friction Limit (μ)", f"{ss.mu_mesa:.2f} G")
        m3.metric("Stability When Stopping", estado_txt(m['desliza']))
        m4.metric("Relative Slip",
                  f"{det_m['deslizamiento_deg_stop']:.4f}° ({m['d_desliza_mm']:.3f} mm)" if m['desliza'] else "0.0000°")
        st.metric("🛡️ Safety Factor When Stopping (μ / g_tan)", fmt_fs(det_m['factor_seguridad_stop']))

        st.divider()
        st.subheader("🌀 Centripetal risk while cruising (constant speed)")
        st.caption(
            "This risk exists ANY time the table rotates at constant speed, even without braking. "
            "It has no equivalent on the linear conveyor."
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Centripetal G @ Fast Speed", f"{det_m['g_cent_fast']:.3f} G")
        mc2.metric("Stability @ Fast", "🔴 RISK" if det_m['se_desliza_cruise_fast'] else "🟢 OK")
        mc3.metric("Centripetal G @ Slow Speed", f"{det_m['g_cent_slow']:.3f} G")
        mc4.metric("Stability @ Slow", "🔴 RISK" if det_m['se_desliza_cruise_slow'] else "🟢 OK")

        if det_m['se_desliza_cruise_fast'] or det_m['se_desliza_cruise_slow']:
            st.error(
                "🔴 **Slip risk while cruising detected.** The centripetal force at this angular speed "
                "and radius exceeds the available friction."
            )
        else:
            st.success("🟢 The centripetal force at both cruise speeds is within the friction limit.")

        with st.expander("🔍 View calculation breakdown — Turntable"):
            st.markdown("### 🧮 Calculation Report (Simulated Values)")

            st.markdown("**Step 0: Sensor Distance → Angle**")
            st.latex(L(
                f"~theta_{{~text{{sensor}}}} = {m['sensor_angle']:.3f}° ~quad ~text{{(from "
                f"{ss.sensor_distance_mm_mesa:.1f}~,~text{{mm}} on rail R={R_sensor:.2f}~,~text{{mm}})}}"
            ))

            st.markdown("**Step 1: Maximum Mechanical Angular Deceleration (Floor)**")
            st.latex(L(
                f"~alpha_{{~text{{mech,max}}}} = ~frac{{~omega_{{~text{{slow}}}}}}{{20~,~text{{ms}}}} = "
                f"~frac{{{ss.speed_slow_mesa:.1f}~,°/s}}{{0.020~,~text{{s}}}} = "
                f"~mathbf{{{det_m['alpha_mechanical_max']:.1f}~,°/s^2}}"
            ))

            st.markdown("**Step 2: Effective Angular Deceleration (RAMP_STOP TABLE capped by the mechanical floor)**")
            st.latex(L(
                f"~alpha_{{~text{{stop}}}} = ~min({ss.ramp_stop_mesa:.1f}~,°/s^2, "
                f"{det_m['alpha_mechanical_max']:.1f}~,°/s^2) = ~mathbf{{{det_m['alpha_stop']:.1f}~,°/s^2}} "
                f"~quad (t~approx{det_m['ramp_stop_real_ms']:.1f}~,~text{{ms}})"
            ))
            st.latex(L(
                f"a_{{~text{{tan~_stop}}}} = ~alpha_{{~text{{stop}}}} ~cdot R_{{~text{{max}}}} "
                f"= {det_m['a_tan_stop']:.2f}~,~text{{mm/s}}^2"
            ))
            st.latex(L(f"g_{{~text{{conv}}}} = ~frac{{{det_m['a_tan_stop']:.2f}}}{{9810}} = ~mathbf{{{m['g_conv']:.3f}~,G}}"))

            st.markdown("**Step 3: Centripetal Component While Cruising**")
            st.latex(L(r"a_{~text{cent}} = ~omega^2 ~cdot R_{~text{max}} ~quad ~text{(in radians)}"))
            st.latex(L(
                f"g_{{~text{{cent,fast}}}} = ~mathbf{{{det_m['g_cent_fast']:.3f}~,G}} ~quad "
                f"g_{{~text{{cent,slow}}}} = ~mathbf{{{det_m['g_cent_slow']:.3f}~,G}}"
            ))

            st.markdown("**Step 4: Slip Criterion When Stopping (tangential only)**")
            if m['desliza']:
                st.error(f"🔴 **SLIPS WHEN STOPPING:** g_conv ({m['g_conv']:.3f} G) > μ ({ss.mu_mesa:.2f} G)")
            else:
                st.success(f"🟢 **STABLE WHEN STOPPING:** g_conv ({m['g_conv']:.3f} G) ≤ μ ({ss.mu_mesa:.2f} G)")

    # ------------------------------------------
    with tab_sens:
        st.subheader("📍 Distance between sensors (mm) — field setup")
        st.caption(
            "Both sensors (Slowdown and Stop) are located near the end of the total angular travel, "
            "mounted on a fixed rail. Enter their physical separation as measured with a tape measure "
            "(sidebar, together with the conversion formula) — the app converts it to the equivalent angle."
        )
        sd1, sd2, sd3 = st.columns(3)
        sd1.metric("Slowdown Sensor Distance TABLE (input)", f"{ss.sensor_distance_mm_mesa:.1f} mm")
        sd2.metric("→ Equivalent Angle Used by the Engine", f"{m['sensor_angle']:.3f}°")
        sd3.metric("Sensor Rail Radius TABLE", f"{R_sensor:.2f} mm")
        st.latex(L(
            f"~theta_{{~text{{sensor}}}} = ~text{{mm~_to~_angle}}({ss.sensor_distance_mm_mesa:.1f}~,~text{{mm}}, "
            f"R={R_sensor:.2f}~,~text{{mm}}) = ~mathbf{{{m['sensor_angle']:.3f}°}}"
        ))
        st.caption(
            f"With this distance, the slowdown sensor triggers at θ = {det_m['angle_sensor_red']:.2f}° "
            f"(t = {m['t_red']:.2f} s) and the stop sensor at θ = {det_m['angle_stop']:.2f}° (t = {m['t_stop']:.2f} s). "
            f"The DECEL ramp needs {det_m['angle_needed_decel']:.2f}° to reach SPEED_AUTO_SLOW."
        )

        st.divider()
        st.subheader("📏 Angular overrun and its conversion to the sensor rail (mm)")
        st.caption(
            "Converts the angular overrun (caused by RAMP_STOP) into the equivalent linear distance on the "
            "sensor rail, so maintenance can reposition the stop sensor with a tape measure."
        )
        overrun_mm_riel = angle_to_mm(m['ang_overrun'], R_sensor, modo_key)
        ov1, ov2, ov3 = st.columns(3)
        ov1.metric("Angular Overrun", f"{m['ang_overrun']:.4f}°")
        ov2.metric(f"Overrun on the Rail (R={R_sensor:.2f}mm)", f"{overrun_mm_riel:.3f} mm")
        ov3.metric("Overrun Time", f"{m['overrun_time']*1000:.0f} ms")

        st.markdown("**🔧 Compensation recommendation (to keep the original stop position):**")
        trigger_angle_original = ss.angle_total_mesa
        trigger_angle_compensado = trigger_angle_original - m['ang_overrun']
        trigger_mm_original = angle_to_mm(trigger_angle_original, R_sensor, modo_key)
        trigger_mm_compensado = angle_to_mm(trigger_angle_compensado, R_sensor, modo_key)

        comp1, comp2 = st.columns(2)
        comp1.metric("Current Stop Sensor Position", f"{trigger_mm_original:.2f} mm")
        comp2.metric("Recommended Compensated Position", f"{trigger_mm_compensado:.2f} mm",
                     delta=f"{(trigger_mm_compensado - trigger_mm_original):.3f} mm (move back)",
                     delta_color="inverse")
        st.info(
            f"💡 For the table to stop at exactly the same point it would reach with "
            f"`RAMP_STOP TABLE = 0`, physically move the **stop sensor** "
            f"**{abs(trigger_mm_compensado - trigger_mm_original):.3f} mm back** along the rail."
        )

    # ------------------------------------------
    with tab_eq:
        st.subheader("📐 Linear equivalence at the critical point (R_max)")
        st.caption(
            f"Converts each angular table parameter into its linear equivalent (mm/s, mm/s²) at the farthest "
            f"point of the part (R_max = {r_max_mesa:.1f} mm), to compare directly against the same "
            f"units on the linear conveyor (Profile A, set up in the Conveyor section)."
        )
        conv_a = simular_conveyor("a")
        st.plotly_chart(figura_equivalencia(m, conv_a))

        eqL1, eqL2 = st.columns(2)
        with eqL1:
            st.markdown("**🔄 Turntable (converted to R_max)**")
            st.metric("SPEED_AUTO_FAST TABLE → mm/s", f"{det_m['v_fast_lineal']:.1f} mm/s")
            st.metric("SPEED_AUTO_SLOW TABLE → mm/s", f"{det_m['v_slow_lineal']:.1f} mm/s")
            st.metric("RAMP_ACCEL TABLE → mm/s²", f"{det_m['a_accel_lineal']:.1f} mm/s²")
            st.metric("RAMP_DECEL TABLE → mm/s²", f"{det_m['a_decel_lineal']:.1f} mm/s²")
        with eqL2:
            st.markdown("**➡️ Linear Conveyor (Profile A, direct reference)**")
            st.metric("SPEED_AUTO_FAST A", f"{ss.speed_fast_a:.1f} mm/s",
                      delta=f"{det_m['v_fast_lineal'] - ss.speed_fast_a:+.1f} mm/s", delta_color="off")
            st.metric("SPEED_AUTO_SLOW A", f"{ss.speed_slow_a:.1f} mm/s",
                      delta=f"{det_m['v_slow_lineal'] - ss.speed_slow_a:+.1f} mm/s", delta_color="off")
            st.metric("RAMP_ACCEL A", f"{ss.accel_a:.1f} mm/s²",
                      delta=f"{det_m['a_accel_lineal'] - ss.accel_a:+.1f} mm/s²", delta_color="off")
            st.metric("RAMP_DECEL A", f"{ss.decel_a:.1f} mm/s²",
                      delta=f"{det_m['a_decel_lineal'] - ss.decel_a:+.1f} mm/s²", delta_color="off")

    # ------------------------------------------
    with tab_fund:
        fundamentos_mesa()


# ==========================================
# PAGE: CONVEYOR
# ==========================================
def sidebar_conveyor():
    sb = st.sidebar
    sb.header("➡️ Parameters — Conveyor")

    sb.subheader("Geometry")
    sb.number_input("Total Conveyor Length (mm)", step=100.0, key="conveyor_length")

    sb.subheader("🔵 Profile A (main)")
    sb.number_input("SPEED_AUTO_FAST A (mm/s)", step=10.0, key="speed_fast_a")
    sb.number_input("SPEED_AUTO_SLOW A (mm/s)", step=10.0, key="speed_slow_a")
    sb.number_input("RAMP_ACCEL A (mm/s²)", step=50.0, key="accel_a")
    sb.number_input("RAMP_DECEL A (mm/s²)", step=50.0, key="decel_a")
    sb.number_input("Slowdown Sensor Distance A (mm)", step=25.0, key="sensor_distance_a")
    sb.number_input(
        "RAMP_STOP A (mm/s²)", step=100.0, min_value=0.0, key="ramp_stop_a",
        help="Deceleration commanded directly by the PLC for the final stop — same kind of parameter as "
             "RAMP_ACCEL / RAMP_DECEL (NOT a time). A mechanical floor (chain backlash, frame flex) "
             "limits the maximum physically achievable deceleration to v_slow / 20 ms. If left at 0, the PLC "
             "is assumed not to limit braking and the system stops as fast as is mechanically possible."
    )
    sb.number_input("Friction Coefficient μ A", step=0.01, min_value=0.01, max_value=1.0, key="mu_a")

    sb.divider()
    sb.checkbox("Compare with Profile B", key="comparar")

    ss = st.session_state
    if ss.comparar and not ss._comparar_prev:
        ss.speed_fast_b = ss.speed_fast_a
        ss.speed_slow_b = ss.speed_slow_a
        ss.accel_b = ss.accel_a
        ss.decel_b = ss.decel_a
        ss.sensor_distance_b = ss.sensor_distance_a
        ss.ramp_stop_b = ss.ramp_stop_a
        ss.mu_b = ss.mu_a
    ss._comparar_prev = ss.comparar

    if ss.comparar:
        sb.subheader("🟣 Profile B (comparison)")
        sb.number_input("SPEED_AUTO_FAST B (mm/s)", step=10.0, key="speed_fast_b")
        sb.number_input("SPEED_AUTO_SLOW B (mm/s)", step=10.0, key="speed_slow_b")
        sb.number_input("RAMP_ACCEL B (mm/s²)", step=50.0, key="accel_b")
        sb.number_input("RAMP_DECEL B (mm/s²)", step=50.0, key="decel_b")
        sb.number_input("Slowdown Sensor Distance B (mm)", step=25.0, key="sensor_distance_b")
        sb.number_input(
            "RAMP_STOP B (mm/s²)", step=100.0, min_value=0.0, key="ramp_stop_b",
            help="Same as RAMP_STOP A — direct deceleration (mm/s²), not a time. 0 = use the mechanical maximum."
        )
        sb.number_input("Friction Coefficient μ B", step=0.01, min_value=0.01, max_value=1.0, key="mu_b")


def figura_movimiento_conveyor(a, b=None):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.46, 0.54], vertical_spacing=0.08,
        subplot_titles=("Conveyor velocity (mm/s)",
                        "Acceleration on the part (G) — green zone = within μ"),
    )
    perfiles = [("A", a, COL_A, "solid")] + ([("B", b, COL_B, "dashdot")] if b is not None else [])

    g_abs_max = 0.0
    for nombre, s, color, dash in perfiles:
        det = s["det"]
        idx = indices_grafica(det["fase"])
        nombres = np.array([FASE_NOMBRE[FASES[c]] for c in det["fase"]], dtype=object)
        g = det["a_cmd"] / 9810.0
        g_abs_max = max(g_abs_max, float(np.max(np.abs(g))) if len(g) else 0.0)
        fig.add_trace(go.Scatter(
            x=s["t"][idx], y=s["vel"][idx], mode="lines", name=f"Velocity {nombre}",
            line=dict(color=color, width=3, dash=dash),
            customdata=np.column_stack([s["pos"][idx], nombres[idx]]),
            hovertemplate=f"{nombre}: " + "%{y:.1f} mm/s · x = %{customdata[0]:.1f} mm · %{customdata[1]}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=s["t"][idx], y=g[idx], mode="lines", name=f"Acceleration {nombre}",
            line=dict(color=color, width=2, dash=dash, shape="vh"),
            hovertemplate=f"{nombre}: " + "%{y:.3f} G<extra></extra>",
        ), row=2, col=1)

    # Sensors
    for nombre, s, color, _ in perfiles:
        c_red = COL_SENSOR_RED if nombre == "A" else COL_B
        c_stop = COL_SENSOR_STOP if nombre == "A" else "#6f4a94"
        for fila in (1, 2):
            fig.add_vline(x=s["t_red"], line_width=2, line_dash="dot", line_color=c_red, row=fila, col=1)
            fig.add_vline(x=s["t_stop"], line_width=2, line_dash="dash", line_color=c_stop, row=fila, col=1)
        entrada_leyenda(fig, f"Slowdown Sensor {nombre} · {s['t_red']:.2f}s · {s['det']['pos_sensor_red']:.0f} mm",
                        c_red, "dot", row=1, col=1)
        entrada_leyenda(fig, f"Stop Sensor {nombre} · {s['t_stop']:.2f}s · {s['det']['pos_stop']:.0f} mm",
                        c_stop, "dash", row=1, col=1)

    # Friction limit
    mu_a = a["mu"]
    fig.add_hrect(y0=-mu_a, y1=mu_a, fillcolor=COL_OK, opacity=0.08, line_width=0, layer="below", row=2, col=1)
    for signo in (1, -1):
        fig.add_hline(y=signo * mu_a, line_width=2, line_dash="dash", line_color=COL_SLIP, row=2, col=1)
    entrada_leyenda(fig, f"Friction limit ±μ{' A' if b is not None else ''} = {mu_a:.2f}",
                    COL_SLIP, "dash", row=2, col=1)
    mu_max = mu_a
    if b is not None and abs(b["mu"] - mu_a) > 1e-9:
        mu_max = max(mu_max, b["mu"])
        for signo in (1, -1):
            fig.add_hline(y=signo * b["mu"], line_width=2, line_dash="dot", line_color=COL_B, row=2, col=1)
        entrada_leyenda(fig, f"Friction limit ±μ B = {b['mu']:.2f}", COL_B, "dot", row=2, col=1)

    # Profile A phases
    det_a = a["det"]
    segs = segmentos_fase(a["t"], det_a["fase"])
    t_total = max(float(s["t"][-1]) for _, s, _, _ in perfiles)
    bandas_fase(fig, segs, lambda sg: abs(det_a["a_cmd"][sg["i1"]]) / 9810.0, mu_a, t_total, fila_etiquetas=1)

    # ---- Value label per ramp, all in one row above the curves ----
    g_top = max(g_abs_max, mu_max, 1e-3)
    items = []
    for nombre, s, color, _ in perfiles:
        det = s["det"]
        rampas = [sg for sg in segmentos_fase(s["t"], det["fase"]) if abs(det["a_cmd"][sg["i1"]]) > 0]
        peor = max(rampas, key=lambda sg: abs(det["a_cmd"][sg["i1"]])) if rampas else None
        for sg in rampas:
            a_val = float(det["a_cmd"][sg["i1"]])
            g = a_val / 9810.0
            desliza = abs(g) > s["mu"]
            dur = sg["t1"] - sg["t0"]
            dur_txt = f"{dur * 1000:.0f} ms" if dur < 0.1 else f"{dur:.2f} s"
            titulo = f"{nombre} · {FASE_CORTO[sg['fase']]}" if b is not None else FASE_CORTO[sg["fase"]]
            items.append({
                "x": (sg["t0"] + sg["t1"]) / 2, "y": g, "desliza": desliza, "peor": sg is peor,
                "texto": texto_etiqueta(titulo, f"{g:+.3f} G".replace("-", "−"), desliza,
                                        f"{abs(a_val):.0f} mm/s² · {dur_txt}",
                                        color_titulo=color if b is not None else "#374151"),
            })
    fila_etiquetas(fig, items, y_fila=g_top * 1.6, t_total=t_total)

    v_max = max(max(float(np.max(s["vel"])) for _, s, _, _ in perfiles), 1e-3)
    fig.update_yaxes(title_text="mm/s", range=[-0.04 * v_max, 1.2 * v_max], row=1, col=1)
    fig.update_yaxes(title_text="G", range=[-g_top * 1.15, g_top * 2.1],
                     tickvals=ticks_bonitos(-g_top, g_top, max_ticks=6), row=2, col=1)
    fig.update_xaxes(range=[0, t_total])
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    leyenda_abajo(fig, 800)
    return fig


def tabla_fases_conveyor(s, perfil=None):
    det = s["det"]
    filas = []
    for sg in segmentos_fase(s["t"], det["fase"]):
        a, b = sg["i0"], sg["i1"]
        g = abs(det["a_cmd"][b]) / 9810.0
        fila = {"Profile": perfil} if perfil else {}
        fila.update({
            "Phase": FASE_NOMBRE[sg["fase"]],
            "PLC state": sg["fase"],
            "Start (s)": round(sg["t0"], 3),
            "Duration (s)": round(sg["t1"] - sg["t0"], 3),
            "Travel (mm)": round(float(s["pos"][b] - s["pos"][a]), 2),
            "v start → end (mm/s)": f"{s['vel'][a]:.0f} → {s['vel'][b]:.0f}",
            "Acceleration (G)": round(g, 3),
            "Status": "🔴 Exceeds μ" if g > s["mu"] else "🟢 OK",
        })
        filas.append(fila)
    return filas


def figura_fases_conveyor(a, b=None):
    etiquetas = ["Start-up<br>ACCEL_FAST", "Braking to creep<br>DECEL_TO_SLOW", "Final stop<br>DECEL_TO_STOP"]

    def valores(s):
        return [s["det"]["g_accel"], s["det"]["g_decel"], s["g_conv"]]

    fig = go.Figure()
    ga = valores(a)
    if b is None:
        colores = [COL_SLIP if g > a["mu"] else COL_OK for g in ga]
    else:
        colores = COL_A
    fig.add_trace(go.Bar(x=etiquetas, y=ga, name="Profile A" if b is not None else "Phase acceleration",
                         marker_color=colores,
                         text=[f"{g:.3f} G" for g in ga], textposition="outside", hovertemplate="%{y:.3f} G"))
    g_all = list(ga)
    if b is not None:
        gb = valores(b)
        g_all += gb
        fig.add_trace(go.Bar(x=etiquetas, y=gb, name="Profile B", marker_color=COL_B,
                             text=[f"{g:.3f} G" for g in gb], textposition="outside", hovertemplate="%{y:.3f} G"))
    fig.add_hline(y=a["mu"], line_width=2, line_dash="dash", line_color=COL_SLIP)
    entrada_leyenda(fig, f"Friction limit μ{' A' if b is not None else ''} = {a['mu']:.2f}", COL_SLIP, "dash")
    mu_max = a["mu"]
    if b is not None and abs(b["mu"] - a["mu"]) > 1e-9:
        mu_max = max(mu_max, b["mu"])
        fig.add_hline(y=b["mu"], line_width=2, line_dash="dot", line_color=COL_B)
        entrada_leyenda(fig, f"Friction limit μ B = {b['mu']:.2f}", COL_B, "dot")
    fig.update_layout(
        title="Acceleration per state transition vs μ", barmode="group", height=400,
        yaxis_title="G", yaxis_range=[0, max(max(g_all), mu_max) * 1.3],
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(b=90, t=70),
    )
    return fig


def pagina_conveyor():
    sidebar_conveyor()
    ss = st.session_state
    comparar = ss.comparar
    A = simular_conveyor("a")
    B = simular_conveyor("b") if comparar else None
    det_a = A["det"]
    det_b = B["det"] if comparar else None

    st.title("➡️ Linear conveyor")
    st.caption("Real-time analysis of the velocity profile, positioning and inertial slip of the part.")

    # ---- Configuration alerts ----
    if det_a['insufficient_distance']:
        st.warning(
            f"⚠️ **Profile A — Inconsistent configuration:** the `RAMP_DECEL A` ramp needs "
            f"**{det_a['dist_needed_decel']:.1f} mm** to go from SPEED_AUTO_FAST down to SPEED_AUTO_SLOW, "
            f"but the `Slowdown Sensor Distance A` is only **{ss.sensor_distance_a:.1f} mm**."
        )
    if comparar and det_b['insufficient_distance']:
        st.warning(
            f"⚠️ **Profile B — Inconsistent configuration:** the `RAMP_DECEL B` ramp needs "
            f"**{det_b['dist_needed_decel']:.1f} mm** to go from SPEED_AUTO_FAST down to SPEED_AUTO_SLOW, "
            f"but the `Slowdown Sensor Distance B` is only **{ss.sensor_distance_b:.1f} mm**."
        )
    if det_a['mechanically_capped']:
        st.info(
            f"ℹ️ **Profile A:** configured `RAMP_STOP A` ({ss.ramp_stop_a:.1f} mm/s²) exceeds the "
            f"achievable mechanical maximum ({det_a['a_mechanical_max']:.1f} mm/s², limited by the "
            f"{T_MIN_MECANICO_MS:.0f} ms floor). The mechanical limit was applied: actual deceleration = "
            f"**{det_a['a_stop_conveyor']:.1f} mm/s²** (≈{det_a['ramp_stop_real_ms']:.1f} ms)."
        )
    if comparar and det_b['mechanically_capped']:
        st.info(
            f"ℹ️ **Profile B:** configured `RAMP_STOP B` ({ss.ramp_stop_b:.1f} mm/s²) exceeds the "
            f"achievable mechanical maximum ({det_b['a_mechanical_max']:.1f} mm/s²). The mechanical limit was applied: "
            f"actual deceleration = **{det_b['a_stop_conveyor']:.1f} mm/s²** (≈{det_b['ramp_stop_real_ms']:.1f} ms)."
        )

    # ---- Summary ----
    def fila_resumen(s, etiqueta, ref=None):
        d = s["det"]
        pos_final = s["pos"][-1] + s["d_desliza"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"Status {etiqueta} (most critical phase)", estado_txt(d['se_desliza_alguna_fase']))
        c2.metric(f"Most critical phase {etiqueta}", f"{d['g_max_fase']:.3f} G", help=d['peor_fase_label'],
                  delta=d['peor_fase_label'].split(" (")[0], delta_color="off")
        c3.metric(f"Safety factor {etiqueta}", fmt_fs(d['factor_seguridad_fase']))
        if ref is None:
            c4.metric(f"Cycle time {etiqueta}", f"{s['t'][-1]:.2f} s")
            c5.metric(f"Final part position {etiqueta}", f"{pos_final:.2f} mm")
        else:
            c4.metric(f"Cycle time {etiqueta}", f"{s['t'][-1]:.2f} s",
                      delta=f"{(s['t'][-1] - ref['t'][-1]):.2f} s vs A", delta_color="inverse")
            ref_pos = ref["pos"][-1] + ref["d_desliza"]
            c5.metric(f"Final part position {etiqueta}", f"{pos_final:.2f} mm",
                      delta=f"{pos_final - ref_pos:.2f} mm vs A", delta_color="off")

    with st.container(border=True):
        st.markdown("**Cycle summary**")
        fila_resumen(A, "A")
        if comparar:
            fila_resumen(B, "B", ref=A)

    tab_mov, tab_slip, tab_ab, tab_fund = st.tabs([
        "📈 Motion", "⚠️ Slip", "⚖️ Profile A vs B", "📚 Fundamentals",
    ])

    # ------------------------------------------
    with tab_mov:
        st.plotly_chart(figura_movimiento_conveyor(A, B))
        st.caption(
            "Labels: acceleration of each ramp in G, with the effective acceleration (mm/s²) and its duration — "
            "green = within μ, red = exceeds μ, thick border = most critical phase. While cruising the acceleration "
            "is 0 G. Bands: Profile A PLC phases. Hover over the velocity curve to see the position x."
        )

        st.subheader("🎯 Positioning & cycle time (Profile A)")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Conveyor Overrun", f"{A['overrun']:.2f} mm")
        col6.metric("Overrun Time", f"{A['overrun_time']*1000:.0f} ms")
        col7.metric("Final Part Position", f"{(A['pos'][-1] + A['d_desliza']):.2f} mm")
        col8.metric("Total Motion Time", f"{A['t'][-1]:.2f} s")

        st.subheader("🧭 Cycle phases")
        filas = tabla_fases_conveyor(A, "A" if comparar else None)
        if comparar:
            filas += tabla_fases_conveyor(B, "B")
        st.dataframe(filas, hide_index=True)

    # ------------------------------------------
    with tab_slip:
        st.plotly_chart(figura_fases_conveyor(A, B))

        st.subheader("Acceleration in G per state transition (Profile A)")
        st.caption(
            "The linear conveyor has three speed transitions where the part experiences acceleration: "
            "start-up (ACCEL_FAST), braking to creep speed (DECEL_TO_SLOW), and the final stop "
            "(DECEL_TO_STOP). Each one is checked against μ separately."
        )
        fa1, fa2, fa3 = st.columns(3)
        fa1.metric("ACCEL_FAST (start-up)", f"{det_a['g_accel']:.3f} G",
                   delta="🔴 SLIP" if det_a['se_desliza_accel'] else "🟢 OK", delta_color="off")
        fa2.metric("DECEL_TO_SLOW (braking to creep)", f"{det_a['g_decel']:.3f} G",
                   delta="🔴 SLIP" if det_a['se_desliza_decel'] else "🟢 OK", delta_color="off")
        fa3.metric("DECEL_TO_STOP (final stop)", f"{A['g_conv']:.3f} G",
                   delta="🔴 SLIP" if det_a['se_desliza_stop_fase'] else "🟢 OK", delta_color="off")

        fb1, fb2 = st.columns(2)
        fb1.metric("🔺 Most Critical Phase", det_a['peor_fase_label'].split(" (")[0], help=det_a['peor_fase_label'])
        fb2.metric("🛡️ Safety Factor of the Most Critical Phase", fmt_fs(det_a['factor_seguridad_fase']))

        if det_a['se_desliza_alguna_fase'] and not det_a['se_desliza_stop_fase']:
            st.warning(
                f"⚠️ **Difference detected:** the final stop is STABLE, but **{det_a['peor_fase_label']}** "
                f"reaches {det_a['g_max_fase']:.3f} G, which **exceeds** μ ({ss.mu_a:.2f} G)."
            )
        elif det_a['se_desliza_alguna_fase']:
            st.error(f"🔴 At least one phase exceeds μ — the most critical is **{det_a['peor_fase_label']}**.")
        else:
            st.success("🟢 All three speed transitions are within the friction limit.")

        st.divider()
        st.subheader("📊 Inertia & part slip at the final stop (Profile A)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Conveyor Deceleration", f"{A['g_conv']:.3f} G")
        col2.metric("Friction Limit (μ)", f"{ss.mu_a:.2f} G")
        col3.metric("Load Stability", estado_txt(A['desliza']))
        col4.metric("Relative Part Slip",
                    f"{A['d_desliza']:.3f} mm ({A['d_desliza']*1000:.0f} µm)" if A['desliza'] else "0.000 mm")

        col_s1, col_s2 = st.columns(2)
        col_s1.metric("🛡️ Safety Factor (μ / g_conv)", fmt_fs(det_a['factor_seguridad']))
        col_s2.metric("🎯 Minimum μ required to NOT slip", f"{det_a['mu_minimo_requerido']:.3f}")

        titulo = ("🔍 View step-by-step calculation breakdown (Profile A)" if A['desliza']
                  else "ℹ️ View stability & deceleration math (Profile A)")
        with st.expander(titulo):
            st.markdown("### 🧮 Live Calculation Breakdown (Simulated Values)")

            st.markdown("**Step 1: Mechanical Maximum Deceleration (Floor Limit)**")
            st.latex(L(
                f"a_{{~text{{mech,max}}}} = ~frac{{v_{{~text{{slow}}}}}}{{20~,~text{{ms}}}} = "
                f"~frac{{{ss.speed_slow_a:.1f}~,~text{{mm/s}}}}{{0.020~,~text{{s}}}} = "
                f"~mathbf{{{det_a['a_mechanical_max']:.1f}~,~text{{mm/s}}^2}}"
            ))

            st.markdown("**Step 2: Effective Stop Deceleration (RAMP_STOP capped by the mechanical limit)**")
            st.latex(L(
                f"a_{{~text{{stop}}}} = ~min({ss.ramp_stop_a:.1f}~,~text{{mm/s}}^2, "
                f"{det_a['a_mechanical_max']:.1f}~,~text{{mm/s}}^2) = ~mathbf{{{det_a['a_stop_conveyor']:.1f}~,~text{{mm/s}}^2}} "
                f"~quad (t~approx{det_a['ramp_stop_real_ms']:.1f}~,~text{{ms}})"
            ))
            st.latex(L(
                f"g_{{~text{{conv}}}} = ~frac{{{det_a['a_stop_conveyor']:.1f}~,~text{{mm/s}}^2}}"
                f"{{9810~,~text{{mm/s}}^2}} = ~mathbf{{{A['g_conv']:.3f}~,~text{{G}}}}"
            ))

            st.markdown("**Step 3: Maximum Allowable Friction Acceleration**")
            st.latex(L(
                f"a_{{~text{{max~_piece}}}} = ~mu ~cdot g = {ss.mu_a:.2f} ~cdot 9810~,~text{{mm/s}}^2 "
                f"= {det_a['a_max_pieza']:.2f}~,~text{{mm/s}}^2 ~quad (~mu = ~mathbf{{{ss.mu_a:.2f}~,~text{{G}}}})"
            ))

            st.markdown("**Step 4: Slip Decision Criterion**")
            if A['desliza']:
                st.error(f"🔴 **SLIP DETECTED:** g_conv ({A['g_conv']:.3f} G) > μ ({ss.mu_a:.2f} G).")
                st.markdown("**Step 5: Relative Slip Distance (Δd)**")
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
                    f"~mathbf{{{A['d_desliza']:.3f}~,~text{{mm}}}}"
                ))
            else:
                st.success(f"🟢 **STABLE LOAD:** g_conv ({A['g_conv']:.3f} G) ≤ μ ({ss.mu_a:.2f} G).")

    # ------------------------------------------
    with tab_ab:
        if not comparar:
            st.info("Turn on **Compare with Profile B** in the sidebar to compare two PLC configurations. "
                    "Profile B starts as a copy of Profile A.")
        else:
            st.subheader("Acceleration per state transition — Profile B")
            st.caption("Deltas are B − A (red = B demands more friction than A).")
            fa1b, fa2b, fa3b = st.columns(3)
            fa1b.metric("ACCEL_FAST B", f"{det_b['g_accel']:.3f} G",
                        delta=f"{(det_b['g_accel'] - det_a['g_accel']):.3f} G", delta_color="inverse")
            fa2b.metric("DECEL_TO_SLOW B", f"{det_b['g_decel']:.3f} G",
                        delta=f"{(det_b['g_decel'] - det_a['g_decel']):.3f} G", delta_color="inverse")
            fa3b.metric("DECEL_TO_STOP B", f"{B['g_conv']:.3f} G",
                        delta=f"{(B['g_conv'] - A['g_conv']):.3f} G", delta_color="inverse")
            st.caption(f"Most critical phase in B: **{det_b['peor_fase_label']}** ({det_b['g_max_fase']:.3f} G)")

            st.divider()
            st.subheader("⚖️ Profile A vs Profile B — final stop & positioning")
            c_b1, c_b2, c_b3, c_b4 = st.columns(4)
            c_b1.metric("Deceleration Profile B", f"{B['g_conv']:.3f} G",
                        delta=f"{(B['g_conv'] - A['g_conv']):.3f} G", delta_color="inverse")
            c_b2.metric("Part Slip Profile B", f"{B['d_desliza']:.3f} mm",
                        delta=f"{(B['d_desliza'] - A['d_desliza']):.3f} mm", delta_color="inverse")
            c_b3.metric("Overrun Profile B", f"{B['overrun']:.2f} mm",
                        delta=f"{(B['overrun'] - A['overrun']):.2f} mm", delta_color="inverse")
            c_b4.metric("Overrun Time Profile B", f"{B['overrun_time']*1000:.0f} ms",
                        delta=f"{(B['overrun_time'] - A['overrun_time'])*1000:.0f} ms", delta_color="inverse")

            c_b5, c_b6 = st.columns(2)
            c_b5.metric("Final Part Position B", f"{(B['pos'][-1] + B['d_desliza']):.2f} mm",
                        delta=f"{(B['pos'][-1] + B['d_desliza']) - (A['pos'][-1] + A['d_desliza']):.2f} mm",
                        delta_color="inverse")
            c_b6.metric("Cycle Time Profile B", f"{B['t'][-1]:.2f} s",
                        delta=f"{(B['t'][-1] - A['t'][-1]):.2f} s", delta_color="inverse")

    # ------------------------------------------
    with tab_fund:
        fundamentos_conveyor()


# ==========================================
# MATHEMATICAL & PHYSICAL BACKGROUND
# ==========================================
def fundamentos_comunes(unidad_a, simbolo):
    """Shared section: mechanical floor (maximum achievable deceleration)."""
    st.subheader("1. Mechanical Elasticity Floor (Maximum Achievable Deceleration)")
    st.markdown(L(
        f"`RAMP_STOP` is a **directly commanded deceleration** ({unidad_a}), consistent with "
        "`RAMP_ACCEL`/`RAMP_DECEL` — it is NOT a time value. Mechanical compliance imposes a minimum "
        "stopping time of $T_{~text{min}} = 20~,~text{ms}$, which translates into a maximum physically "
        "achievable deceleration:"
    ))
    if simbolo == "a":
        st.latex(L(r"a_{~text{mech,max}} = ~frac{v_{~text{slow}}}{T_{~text{min}}}"))
    else:
        st.latex(L(r"~alpha_{~text{mech,max}} = ~frac{~omega_{~text{slow}}}{T_{~text{min}}} ~quad [°/s^2]"))


def fundamentos_conveyor():
    st.header("📐 Physics Engine & Kinematic Formulas — Conveyor")
    st.markdown(
        "Analytical models used to simulate conveyor dynamics, deceleration forces, friction boundaries "
        "and the inertial displacement of the transported part."
    )
    fundamentos_comunes("mm/s²", "a")

    st.subheader("2. Effective Stop Deceleration")
    st.latex(L(
        r"a_{~text{stop}} = ~begin{cases} a_{~text{mech,max}} & ~text{RAMP~_STOP} = 0 \\ "
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
    st.latex(L(r"~text{Safety Factor} = ~frac{~mu}{g_{~text{conv}}} ~qquad ~mu_{~text{min}} = g_{~text{conv}}"))

    st.subheader("6. Sensor Layout: Both Sensors Near the End of Travel")
    st.latex(L(r"P_{~text{stop}} = L_{~text{total}} ~quad P_{~text{reduction}} = L_{~text{total}} - S_{~text{distance}}"))

    st.subheader("7. Sensor Distance Validation")
    st.latex(L(r"d_{~text{needed}} = ~frac{v_{~text{fast}}^2 - v_{~text{slow}}^2}{2 ~cdot ~text{RAMP~_DECEL}}"))

    st.subheader("8. Acceleration per State Transition")
    st.latex(L(r"g_{~text{accel}} = ~frac{~text{RAMP~_ACCEL}}{9810}"))
    st.latex(L(r"g_{~text{decel}} = ~frac{~text{RAMP~_DECEL}}{9810}"))
    st.latex(L(r"g_{~text{stop}} = ~frac{a_{~text{stop}}}{9810}"))
    st.latex(L(r"g_{~text{max~_phase}} = ~max~left(g_{~text{accel}}, g_{~text{decel}}, g_{~text{stop}}~right)"))

    st.subheader("9. State Machine & Numerical Integration")
    st.markdown(L(
        "The PLC is modeled as the sequence `ACCEL_FAST → CRUISE_FAST → DECEL_TO_SLOW → CRUISE_SLOW → "
        "DECEL_TO_STOP`, integrated with a step of $~Delta t = 1~,~text{ms}$. The accumulated position is "
        "what triggers the slowdown and stop sensors:"
    ))
    st.latex(L(r"x_i = x_{i-1} + v_i ~cdot ~Delta t"))


def fundamentos_mesa():
    st.header("📐 Physics Engine & Kinematic Formulas — Turntable")
    st.markdown(
        "Analytical models used to simulate the table rotation, the tangential and centripetal components "
        "felt by the part, and the conversion of angular positions to the sensor rail."
    )
    fundamentos_comunes("°/s²", "alpha")

    st.subheader("2. Effective Angular Deceleration & Tangential Component (Braking)")
    st.latex(L(
        r"~alpha_{~text{stop}} = ~begin{cases} ~alpha_{~text{mech,max}} & ~text{RAMP~_STOP~_TABLE} = 0 \\ "
        r"~min(~text{RAMP~_STOP~_TABLE}, ~alpha_{~text{mech,max}}) & ~text{RAMP~_STOP~_TABLE} > 0 ~end{cases}"
    ))
    st.latex(L(r"a_{~text{tan~_stop}} = ~alpha_{~text{stop,rad}} ~cdot R_{~text{max}} ~quad [~text{mm/s}^2]"))
    st.latex(L(r"g_{~text{conv}} = ~frac{a_{~text{tan~_stop}}}{9810}"))

    st.subheader("3. Static Friction Threshold")
    st.latex(L(r"a_{~text{max~_piece}} = ~mu ~cdot g = ~mu ~cdot 9810~,~text{mm/s}^2"))

    st.subheader("4. Angular-to-Linear Analogy")
    st.latex(L(r"R_{~text{max}} = ~frac{~text{Total Part Length}}{2}"))
    st.latex(L(r"v_{~text{linear}} = ~omega_{~text{rad}} ~cdot R_{~text{max}} ~quad a_{~text{linear}} = ~alpha_{~text{rad}} ~cdot R_{~text{max}}"))

    st.subheader("5. Angular Position via Time Integration")
    st.markdown(L(
        "The rotated angle is the integral of the angular velocity. In the discrete simulation it is the "
        "running sum that triggers the slowdown and stop sensors (its value at each instant shows when "
        "hovering over the velocity chart):"
    ))
    st.latex(L(r"~theta(t) = ~int_0^t ~omega(~tau)~,d~tau ~qquad ~theta_i = ~theta_{i-1} + ~omega_i ~cdot ~Delta t"))

    st.subheader("6. Centripetal Component (Constant-Speed Cruise)")
    st.latex(L(r"a_{~text{cent}} = ~omega_{~text{rad}}^2 ~cdot R_{~text{max}} ~quad [~text{mm/s}^2]"))
    st.latex(L(r"g_{~text{cent}} = ~frac{a_{~text{cent}}}{9810}"))

    st.subheader("7. Combined Resultant Acceleration at Critical Instants")
    st.latex(L(r"a_{~text{resultant}} = ~sqrt{a_{~text{tangential}}^2 + a_{~text{centripetal}}^2}"))

    st.subheader("8. Relative Slip When Stopping")
    st.latex(L(r"d_{~text{piece}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{max~_piece}}} ~quad d_{~text{table}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{tan~_stop}}}"))
    st.latex(L(r"~Delta d = d_{~text{piece}} - d_{~text{table}} ~qquad ~Delta ~theta = ~frac{~Delta d}{R_{~text{max}}}"))

    st.subheader("9. Safety Factor")
    st.latex(L(r"~text{SF} = ~frac{~mu}{g} ~quad ~text{(braking, cruise or resultant)}"))

    st.subheader("10. Sensors: Position & Validation")
    st.latex(L(r"~theta_{~text{stop}} = ~theta_{~text{total}} ~quad ~theta_{~text{reduction}} = ~theta_{~text{total}} - ~theta_{~text{sensor~_dist}}"))
    st.latex(L(r"~theta_{~text{needed}} = ~frac{~omega_{~text{fast}}^2 - ~omega_{~text{slow}}^2}{2 ~cdot ~text{RAMP~_DECEL~_TABLE}}"))

    st.subheader("11. Sensor Rail: Angle ↔ mm Conversion for Maintenance")
    st.latex(L(r"~theta_{~text{sensor}} = ~frac{s_{~text{mm}}}{R_{~text{sensor}}} ~quad ~text{(Arc, radians → degrees)}"))
    st.latex(L(r"~theta_{~text{sensor}} = 2 ~arcsin~left(~frac{s_{~text{mm}}}{2R_{~text{sensor}}}~right) ~quad ~text{(Chord)}"))
    st.latex(L(r"s_{~text{arc}} = R_{~text{sensor}} ~cdot ~theta_{~text{rad}}"))
    st.latex(L(r"s_{~text{chord}} = 2R_{~text{sensor}} ~sin~left(~frac{~theta_{~text{rad}}}{2}~right)"))

    st.subheader("12. Scope Limitation: Free-Sliding Dynamics on a Rotating Frame")
    st.markdown(
        "A fully rigorous simulation of an object sliding freely on a rotating platform requires solving "
        "the motion in a non-inertial rotating reference frame, with Coriolis and Euler pseudo-forces. "
        "This model evaluates the slip threshold, not the subsequent trajectory."
    )


# ==========================================
# NAVIGATION: TWO SECTIONS
# ==========================================
pagina = st.navigation(
    [
        st.Page(pagina_mesa, title="Turntable", icon=":material/360:", url_path="turntable", default=True),
        st.Page(pagina_conveyor, title="Conveyor", icon=":material/conveyor_belt:", url_path="conveyor"),
    ],
    position="top",
)
pagina.run()
