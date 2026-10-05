import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
st.set_page_config(
    page_title="Simulador Cinemático — Mesa Giratoria y Conveyor",
    page_icon="⚙️",
    layout="wide",
)

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
# CONSTANTES
# ==========================================
T_MIN_MECANICO_MS = 20.0
T_MIN_MECANICO_S = T_MIN_MECANICO_MS / 1000.0

# Colores (consistentes en toda la app)
COL_A = "#1f77b4"
COL_B = "#9467bd"
COL_SENSOR_RED = "#ff7f0e"
COL_SENSOR_STOP = "#d62728"
COL_OK = "#2ca02c"
COL_SLIP = "#d62728"
COL_TAN = "#ff7f0e"
COL_CENT = "#17becf"
COL_NEUTRO = "#7f7f7f"

# Estados de la máquina de estados del PLC (mismo orden en ambos motores)
FASES = ["ACCEL_FAST", "CRUISE_FAST", "DECEL_TO_SLOW", "CRUISE_SLOW", "DECEL_TO_STOP", "DONE"]
FASE_CODE = {f: i for i, f in enumerate(FASES)}
FASE_NOMBRE = {
    "ACCEL_FAST": "Arranque",
    "CRUISE_FAST": "Crucero rápido",
    "DECEL_TO_SLOW": "Frenado a creep",
    "CRUISE_SLOW": "Crucero lento (creep)",
    "DECEL_TO_STOP": "Frenado final",
    "DONE": "Detenido",
}
FASE_CORTO = {
    "ACCEL_FAST": "ACCEL",
    "CRUISE_FAST": "CRUCERO",
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
    "modo_conversion_mesa": "Arco (R·θ) — recomendado",
}
# Re-asignar cada valor en cada ejecución hace que los parámetros persistan al cambiar
# de sección (Streamlit descarta el estado de los widgets que no se dibujan en la página activa).
for key, val in defaults.items():
    st.session_state[key] = st.session_state.get(key, val)
if "_comparar_prev" not in st.session_state:
    st.session_state._comparar_prev = False


# ==========================================
# REALISTIC KINEMATIC CALCULATION ENGINE (CONVEYOR LINEAL)
# ==========================================
@st.cache_data(show_spinner=False)
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist, ramp_stop_accel, mu):
    dt = 0.001
    t_max = 30.0
    steps = int(t_max / dt)
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)
    # Registro por paso: estado que gobernó el paso y aceleración comandada (mm/s²)
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
        "ACCEL_FAST (arranque)": g_accel,
        "DECEL_TO_SLOW (frenado a creep)": g_decel,
        "DECEL_TO_STOP (frenado final)": g_conveyor,
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
# ANGULAR KINEMATIC CALCULATION ENGINE (MESA GIRATORIA)
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
    # Registro por paso: estado que gobernó el paso y aceleración angular comandada (°/s²)
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
        "Fin de ACCEL_FAST (ω=ω_fast, tan=accel)": g_res_fin_accel,
        "Inicio de DECEL_TO_SLOW (ω=ω_fast, tan=decel)": g_res_ini_decel,
        "Fin de DECEL_TO_SLOW (ω=ω_slow, tan=decel)": g_res_fin_decel,
        "Inicio de DECEL_TO_STOP (ω=ω_slow, tan=stop)": g_res_ini_stop,
    }
    # Componentes (tangencial, centrípeta) en G de cada instante crítico — para graficar
    componentes = {
        "Fin de ACCEL_FAST (ω=ω_fast, tan=accel)": (a_accel_lineal / 9810.0, g_cent_fast),
        "Inicio de DECEL_TO_SLOW (ω=ω_fast, tan=decel)": (a_decel_lineal / 9810.0, g_cent_fast),
        "Fin de DECEL_TO_SLOW (ω=ω_slow, tan=decel)": (a_decel_lineal / 9810.0, g_cent_slow),
        "Inicio de DECEL_TO_STOP (ω=ω_slow, tan=stop)": (g_conv_mesa, g_cent_slow),
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
# AYUDAS DE PRESENTACIÓN
# ==========================================
def fmt_fs(fs):
    return "∞" if fs == float('inf') else f"{fs:.2f}x"


def estado_txt(desliza):
    return "🔴 DESLIZA" if desliza else "🟢 ESTABLE"


def indices_grafica(fase, paso=5):
    """Submuestrea para graficar sin perder los cambios de fase (las curvas son lineales por tramos)."""
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
    """Tramos contiguos de cada estado del PLC: índice de inicio/fin y tiempos."""
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
        i0 = max(int(s) - 1, 0)  # instante en que arranca la fase
        i1 = int(e) - 1          # último paso gobernado por la fase
        segs.append({"fase": nombre, "i0": i0, "i1": i1, "t0": float(t[i0]), "t1": float(t[i1])})
    return segs


def bandas_fase(fig, segs, g_fase, mu, t_total, fila_etiquetas=2):
    """Sombrea cada fase en ambas filas (rojo si supera μ) y la rotula en la fila indicada."""
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


def entrada_leyenda(fig, nombre, color, dash, **pos):
    """Entrada solo de leyenda (sensores, μ): no ocupa espacio en la gráfica ni afecta los ejes."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nombre, hoverinfo="skip",
                             line=dict(color=color, width=2, dash=dash)), **pos)


def leyenda_abajo(fig, height, legend_y=-0.1):
    fig.update_layout(
        height=height, hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=legend_y, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(b=120, t=70),
    )


# ==========================================
# SIMULACIONES (envoltorios que leen los parámetros de la sesión)
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
    return "Arco (R·θ)" if st.session_state.modo_conversion_mesa.startswith("Arco") else "Cuerda (2R·sin(θ/2))"


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
    """G tangencial, centrípeta y resultante en R_max para cada instante de la simulación."""
    r = m["r_max"]
    g_tan = np.abs(np.radians(m["det"]["alpha_cmd"])) * r / 9810.0
    g_cent = (np.radians(m["omega"]) ** 2) * r / 9810.0
    g_res = np.sqrt(g_tan ** 2 + g_cent ** 2)
    return g_tan, g_cent, g_res


# ==========================================
# PÁGINA: MESA GIRATORIA
# ==========================================
def sidebar_mesa():
    sb = st.sidebar
    sb.header("🔄 Parámetros — Mesa giratoria")

    sb.subheader("Cinemática (PLC)")
    sb.number_input("SPEED_AUTO_FAST MESA (°/s)", step=5.0, key="speed_fast_mesa")
    sb.number_input("SPEED_AUTO_SLOW MESA (°/s)", step=1.0, key="speed_slow_mesa")
    sb.number_input("RAMP_ACCEL MESA (°/s²)", step=10.0, key="accel_mesa")
    sb.number_input("RAMP_DECEL MESA (°/s²)", step=10.0, key="decel_mesa")
    sb.number_input(
        "RAMP_STOP MESA (°/s²)", step=25.0, min_value=0.0, key="ramp_stop_mesa",
        help="Deceleración angular directamente comandada por el PLC para el frenado final — mismo tipo de "
             "parámetro que RAMP_ACCEL/RAMP_DECEL MESA (NO es un tiempo). Un piso mecánico limita la deceleración "
             "angular máxima físicamente alcanzable a ω_slow / 20ms. Si se deja en 0, se asume el máximo mecánico."
    )
    sb.number_input("Ángulo Total de Giro MESA (°)", step=15.0, key="angle_total_mesa",
                    help="Equivalente angular de 'Largo Total Conveyor' — abierto")

    sb.subheader("Sensores sobre el riel")
    sb.number_input("Radio Riel Sensores MESA (mm)", step=5.0, min_value=1.0, key="radio_sensor_mesa",
                    help="Radio físico donde están montados los sensores de reducción/paro sobre el riel — "
                         "distinto del radio de la pieza")
    sb.radio(
        "Fórmula de conversión Ángulo ↔ mm",
        ["Arco (R·θ) — recomendado", "Cuerda (2R·sin(θ/2))"],
        key="modo_conversion_mesa",
        help="Arco = distancia exacta si el riel sigue la curvatura a ese radio (caso típico). "
             "Cuerda = distancia recta entre dos puntos del círculo (útil si el riel es perfectamente recto)."
    )
    sb.number_input(
        "Distancia Sensor Reducción MESA (mm)", step=5.0, min_value=0.0, key="sensor_distance_mm_mesa",
        help="Separación física, medida sobre el riel, entre el Sensor de Reducción y el Sensor de Paro. "
             "Ambos sensores están ubicados cerca del final del recorrido angular total."
    )
    _ang = mm_to_angle(st.session_state.sensor_distance_mm_mesa, st.session_state.radio_sensor_mesa,
                       modo_conversion_key())
    sb.caption(f"↳ Equivalente angular: **{_ang:.3f}°** (modo: {modo_conversion_key()})")

    sb.subheader("Pieza")
    sb.number_input("Longitud Total de Pieza MESA (mm)", step=10.0, min_value=1.0, key="pieza_longitud_mesa",
                    help="La pieza se posiciona centrada en el eje de giro. R_max = Longitud / 2 "
                         "(usado SOLO para el análisis de deslizamiento)")
    sb.number_input("Coeficiente Fricción μ MESA", step=0.01, min_value=0.01, max_value=1.0, key="mu_mesa")


def figura_movimiento_mesa(m):
    det = m["det"]
    t, omega, theta = m["t"], m["omega"], m["theta"]
    g_tan, g_cent, g_res = series_g_mesa(m)
    idx = indices_grafica(det["fase"])
    nombres = np.array([FASE_NOMBRE[FASES[c]] for c in det["fase"]], dtype=object)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42], vertical_spacing=0.08,
        subplot_titles=("Velocidad angular de la mesa (°/s)",
                        f"Aceleración sobre la pieza en R_max = {m['r_max']:.0f} mm (G) — zona verde = dentro de μ"),
    )

    fig.add_trace(go.Scatter(
        x=t[idx], y=omega[idx], mode="lines", name="Velocidad angular ω",
        line=dict(color=COL_A, width=3),
        customdata=np.column_stack([theta[idx], nombres[idx]]),
        hovertemplate="ω = %{y:.2f} °/s · θ = %{customdata[0]:.2f}° · %{customdata[1]}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t[idx], y=g_tan[idx], mode="lines", name="Tangencial (rampa)",
        line=dict(color=COL_TAN, width=1.5, shape="vh"), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t[idx], y=g_cent[idx], mode="lines", name="Centrípeta (ω²·R)",
        line=dict(color=COL_CENT, width=1.5), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t[idx], y=g_res[idx], mode="lines", name="Resultante",
        line=dict(color="#444444", width=2.5), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)

    mu = m["mu"]
    g_top = max(float(np.max(g_res)) if len(g_res) else 0.0, mu)
    fig.add_hrect(y0=0, y1=mu, fillcolor=COL_OK, opacity=0.08, line_width=0, layer="below", row=2, col=1)
    fig.add_hline(y=mu, line_width=2, line_dash="dash", line_color=COL_SLIP, row=2, col=1)
    entrada_leyenda(fig, f"Límite de fricción μ = {mu:.2f}", COL_SLIP, "dash", row=2, col=1)

    i_max = int(np.argmax(g_res)) if len(g_res) else 0
    fig.add_annotation(
        x=t[i_max], y=g_res[i_max], xref="x2", yref="y2", text=f"Máx. {g_res[i_max]:.3f} G",
        showarrow=True, arrowhead=2, ax=-55, ay=-20,
        font=dict(size=10, color=COL_SLIP if g_res[i_max] > mu else COL_OK),
    )

    for fila in (1, 2):
        fig.add_vline(x=m["t_red"], line_width=2, line_dash="dot", line_color=COL_SENSOR_RED, row=fila, col=1)
        fig.add_vline(x=m["t_stop"], line_width=2, line_dash="dash", line_color=COL_SENSOR_STOP, row=fila, col=1)
    entrada_leyenda(fig, f"Sensor Reducción · {m['t_red']:.2f}s · {m['theta_red']:.2f}°",
                    COL_SENSOR_RED, "dot", row=1, col=1)
    entrada_leyenda(fig, f"Sensor Paro · {m['t_stop']:.2f}s · {m['theta_stop']:.2f}°",
                    COL_SENSOR_STOP, "dash", row=1, col=1)

    segs = segmentos_fase(t, det["fase"])

    def g_fase(sg):
        a, b = sg["i0"], sg["i1"]
        tan = abs(np.radians(det["alpha_cmd"][b])) * m["r_max"] / 9810.0
        cent = float(np.max(g_cent[a:b + 1]))
        return float(np.sqrt(tan ** 2 + cent ** 2))

    bandas_fase(fig, segs, g_fase, mu, float(t[-1]))

    fig.update_yaxes(title_text="°/s", row=1, col=1)
    fig.update_yaxes(title_text="G", range=[0, g_top * 1.4], row=2, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
    leyenda_abajo(fig, 660)
    return fig


def tabla_fases_mesa(m):
    det = m["det"]
    _, g_cent, _ = series_g_mesa(m)
    filas = []
    for sg in segmentos_fase(m["t"], det["fase"]):
        a, b = sg["i0"], sg["i1"]
        tan = abs(np.radians(det["alpha_cmd"][b])) * m["r_max"] / 9810.0
        cent = float(np.max(g_cent[a:b + 1]))
        res = float(np.sqrt(tan ** 2 + cent ** 2))
        filas.append({
            "Fase": FASE_NOMBRE[sg["fase"]],
            "Estado PLC": sg["fase"],
            "Inicio (s)": round(sg["t0"], 3),
            "Duración (s)": round(sg["t1"] - sg["t0"], 3),
            "Giro (°)": round(float(m["theta"][b] - m["theta"][a]), 3),
            "ω inicio → fin (°/s)": f"{m['omega'][a]:.1f} → {m['omega'][b]:.1f}",
            "G tangencial": round(tan, 3),
            "G centrípeta máx.": round(cent, 3),
            "G resultante máx.": round(res, 3),
            "Estado": "🔴 Supera μ" if res > m["mu"] else "🟢 OK",
        })
    return filas


def figura_instantes_mesa(m):
    det = m["det"]
    mu = m["mu"]
    etiquetas = [k.split(" (")[0].replace(" de ", "<br>") for k in det["resultantes"]]
    tan = [c[0] for c in det["componentes"].values()]
    cent = [c[1] for c in det["componentes"].values()]
    res = list(det["resultantes"].values())

    fig = go.Figure()
    fig.add_trace(go.Bar(x=etiquetas, y=tan, name="Tangencial", marker_color=COL_TAN, opacity=0.55,
                         hovertemplate="%{y:.3f} G"))
    fig.add_trace(go.Bar(x=etiquetas, y=cent, name="Centrípeta", marker_color=COL_CENT, opacity=0.55,
                         hovertemplate="%{y:.3f} G"))
    fig.add_trace(go.Bar(x=etiquetas, y=res, name="Resultante",
                         marker_color=[COL_SLIP if g > mu else COL_OK for g in res],
                         text=[f"{g:.3f} G" for g in res], textposition="outside",
                         hovertemplate="%{y:.3f} G"))
    fig.add_hline(y=mu, line_width=2, line_dash="dash", line_color=COL_SLIP)
    entrada_leyenda(fig, f"Límite de fricción μ = {mu:.2f}", COL_SLIP, "dash")
    fig.update_layout(
        title="Instantes críticos: componentes y resultante vs μ", barmode="group", height=420,
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
                             name=f"Mesa — velocidad lineal en R_max ({m['r_max']:.0f} mm)",
                             line=dict(color=COL_A, width=3), hovertemplate="%{y:.1f} mm/s"))
    fig.add_trace(go.Scatter(x=conv_a["t"][idx_c], y=conv_a["vel"][idx_c], mode="lines",
                             name="Conveyor — Perfil A", line=dict(color=COL_NEUTRO, width=2, dash="dash"),
                             hovertemplate="%{y:.1f} mm/s"))
    fig.update_layout(title="Velocidad lineal (mm/s) vs tiempo — mesa en R_max vs conveyor",
                      xaxis_title="Tiempo (s)", yaxis_title="mm/s")
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

    st.title("🔄 Mesa giratoria")
    st.caption(
        f"Pieza centrada en el eje de giro. R_max (deslizamiento) = Longitud/2 = {r_max_mesa:.1f} mm. "
        f"⚠️ Distinto del Radio Riel Sensores ({R_sensor:.2f} mm), que solo se usa "
        f"para convertir posiciones de sensores entre grados y mm."
    )

    # ---- Alertas de configuración ----
    if det_m['insufficient_angle']:
        st.warning(
            f"⚠️ **Configuración inconsistente:** `RAMP_DECEL MESA` necesita "
            f"**{det_m['angle_needed_decel']:.1f}°** para bajar de SPEED_AUTO_FAST a SPEED_AUTO_SLOW, "
            f"pero la `Distancia Sensor Reducción MESA` ({ss.sensor_distance_mm_mesa:.1f} mm ≈ "
            f"{m['sensor_angle']:.1f}°) es insuficiente."
        )
    if det_m['mechanically_capped']:
        st.info(
            f"ℹ️ `RAMP_STOP MESA` configurado ({ss.ramp_stop_mesa:.1f} °/s²) excede el "
            f"máximo mecánico alcanzable ({det_m['alpha_mechanical_max']:.1f} °/s², limitado por el piso de "
            f"{T_MIN_MECANICO_MS:.0f} ms). Se aplicó el límite mecánico: deceleración angular real = "
            f"**{det_m['alpha_stop']:.1f} °/s²** (≈{det_m['ramp_stop_real_ms']:.1f} ms)."
        )

    # ---- Resumen ----
    with st.container(border=True):
        st.markdown("**Resumen del ciclo**")
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Estado (peor caso combinado)", estado_txt(det_m['se_desliza_resultante']),
                  help="Criterio más completo: tangencial + centrípeta en los instantes críticos.")
        v2.metric("Aceleración máx. sobre la pieza", f"{det_m['g_res_max']:.3f} G",
                  help=f"Ocurre en: {det_m['peor_caso_label']}")
        v3.metric("Factor de seguridad", fmt_fs(det_m['factor_seguridad_resultante']))
        v4.metric("Tiempo de ciclo", f"{t_m[-1]:.2f} s")
        v5.metric("Overrun angular", f"{m['ang_overrun']:.3f}°")

    tab_mov, tab_slip, tab_sens, tab_eq, tab_fund = st.tabs([
        "📈 Movimiento", "⚠️ Deslizamiento", "📍 Sensores y ajuste", "🔁 Equivalencia con conveyor",
        "📚 Fundamentos",
    ])

    # ------------------------------------------
    with tab_mov:
        st.plotly_chart(figura_movimiento_mesa(m))
        st.caption(
            "Bandas: fases del PLC (gris = rampa, rojo = la aceleración resultante supera μ). "
            "Pasa el cursor sobre la curva de velocidad para ver el ángulo acumulado θ y la fase en cada instante."
        )

        st.subheader("🎯 Posicionamiento y tiempo de ciclo")
        mp1, mp2, mp3, mp4, mp5 = st.columns(5)
        mp1.metric("Overrun Angular", f"{m['ang_overrun']:.3f}°")
        mp2.metric("Overrun Time", f"{m['overrun_time']*1000:.0f} ms")
        mp3.metric("Ángulo Final de la Mesa", f"{theta_m[-1]:.3f}°",
                   help=f"Debe ser el Ángulo Total de Giro ({ss.angle_total_mesa:.1f}°) más el overrun.")
        mp4.metric("Posición Final Pieza", f"{(theta_m[-1] + det_m['deslizamiento_deg_stop']):.3f}°")
        mp5.metric("Tiempo Total de Ciclo", f"{t_m[-1]:.2f} s")

        st.subheader("🧭 Fases del ciclo")
        st.dataframe(tabla_fases_mesa(m), hide_index=True)

    # ------------------------------------------
    with tab_slip:
        st.plotly_chart(figura_instantes_mesa(m))

        st.subheader("Aceleración resultante en los instantes críticos (tangencial + centrípeta)")
        st.caption(
            "En los extremos entre fases, la pieza siente AMBAS componentes simultáneamente. El criterio "
            "'ESTABLE/DESLIZA' del frenado final (más abajo) usa solo la componente tangencial — este "
            "análisis muestra el peor caso combinado real."
        )
        res_cols = st.columns(4)
        for col, (label, g_val) in zip(res_cols, det_m['resultantes'].items()):
            is_worst = (label == det_m['peor_caso_label'])
            col.metric(label.split(" (")[0], f"{g_val:.3f} G",
                       delta="⚠️ PEOR CASO" if is_worst else None, delta_color="off", help=label)

        rc1, rc2 = st.columns(2)
        rc1.metric("🔺 Peor Caso Combinado (Resultante Máxima)", f"{det_m['g_res_max']:.3f} G",
                   help=f"Ocurre en: {det_m['peor_caso_label']}")
        rc2.metric("🛡️ Factor de Seguridad Resultante (μ / g_res_max)", fmt_fs(det_m['factor_seguridad_resultante']))

        if det_m['se_desliza_resultante'] and not m['desliza']:
            st.warning(
                f"⚠️ **Diferencia detectada:** el criterio actual (solo tangencial) marca **ESTABLE**, pero el "
                f"criterio resultante indica que en **{det_m['peor_caso_label']}** la fuerza total "
                f"({det_m['g_res_max']:.3f} G) **supera** μ ({ss.mu_mesa:.2f} G)."
            )
        elif det_m['se_desliza_resultante']:
            st.error(f"🔴 El criterio resultante confirma deslizamiento — peor caso en **{det_m['peor_caso_label']}**.")
        else:
            st.success("🟢 Incluso considerando el peor caso combinado, la pieza permanece estable en todo el ciclo.")

        st.divider()
        st.subheader("Frenado final (componente tangencial)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Deceleración Tangencial (frenado)", f"{m['g_conv']:.3f} G")
        m2.metric("Límite de Fricción (μ)", f"{ss.mu_mesa:.2f} G")
        m3.metric("Estabilidad al Frenar", estado_txt(m['desliza']))
        m4.metric("Deslizamiento Relativo",
                  f"{det_m['deslizamiento_deg_stop']:.4f}° ({m['d_desliza_mm']:.3f} mm)" if m['desliza'] else "0.0000°")
        st.metric("🛡️ Factor de Seguridad al Frenar (μ / g_tan)", fmt_fs(det_m['factor_seguridad_stop']))

        st.divider()
        st.subheader("🌀 Riesgo centrípeto en crucero (velocidad constante)")
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
                "y radio supera la fricción disponible."
            )
        else:
            st.success("🟢 La fuerza centrípeta en ambas velocidades de crucero está dentro del límite de fricción.")

        with st.expander("🔍 Ver desglose de cálculo — Mesa giratoria"):
            st.markdown("### 🧮 Memoria de Cálculo (Valores Simulados)")

            st.markdown("**Paso 0: Distancia Sensor → Ángulo**")
            st.latex(L(
                f"~theta_{{~text{{sensor}}}} = {m['sensor_angle']:.3f}° ~quad ~text{{(desde "
                f"{ss.sensor_distance_mm_mesa:.1f}~,~text{{mm}} sobre riel R={R_sensor:.2f}~,~text{{mm}})}}"
            ))

            st.markdown("**Paso 1: Deceleración Angular Mecánica Máxima (Piso)**")
            st.latex(L(
                f"~alpha_{{~text{{mech,max}}}} = ~frac{{~omega_{{~text{{slow}}}}}}{{20~,~text{{ms}}}} = "
                f"~frac{{{ss.speed_slow_mesa:.1f}~,°/s}}{{0.020~,~text{{s}}}} = "
                f"~mathbf{{{det_m['alpha_mechanical_max']:.1f}~,°/s^2}}"
            ))

            st.markdown("**Paso 2: Deceleración Angular Efectiva (RAMP_STOP MESA limitado por el piso mecánico)**")
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

            st.markdown("**Paso 3: Componente Centrípeta en Crucero**")
            st.latex(L(r"a_{~text{cent}} = ~omega^2 ~cdot R_{~text{max}} ~quad ~text{(en radianes)}"))
            st.latex(L(
                f"g_{{~text{{cent,fast}}}} = ~mathbf{{{det_m['g_cent_fast']:.3f}~,G}} ~quad "
                f"g_{{~text{{cent,slow}}}} = ~mathbf{{{det_m['g_cent_slow']:.3f}~,G}}"
            ))

            st.markdown("**Paso 4: Criterio de Deslizamiento al Frenar (tangencial simple)**")
            if m['desliza']:
                st.error(f"🔴 **DESLIZA AL FRENAR:** g_conv ({m['g_conv']:.3f} G) > μ ({ss.mu_mesa:.2f} G)")
            else:
                st.success(f"🟢 **ESTABLE AL FRENAR:** g_conv ({m['g_conv']:.3f} G) ≤ μ ({ss.mu_mesa:.2f} G)")

    # ------------------------------------------
    with tab_sens:
        st.subheader("📍 Distancia entre sensores (mm) — configuración de campo")
        st.caption(
            "Ambos sensores (Reducción y Paro) están ubicados cerca del final del recorrido angular total, "
            "montados sobre un riel fijo. Ingresa su separación física tal como se mide con cinta métrica "
            "(barra lateral, junto con la fórmula de conversión) — la app la convierte al ángulo equivalente."
        )
        sd1, sd2, sd3 = st.columns(3)
        sd1.metric("Distancia Sensor Reducción MESA (entrada)", f"{ss.sensor_distance_mm_mesa:.1f} mm")
        sd2.metric("→ Ángulo Equivalente Usado por el Motor", f"{m['sensor_angle']:.3f}°")
        sd3.metric("Radio Riel Sensores MESA", f"{R_sensor:.2f} mm")
        st.latex(L(
            f"~theta_{{~text{{sensor}}}} = ~text{{mm~_to~_angle}}({ss.sensor_distance_mm_mesa:.1f}~,~text{{mm}}, "
            f"R={R_sensor:.2f}~,~text{{mm}}) = ~mathbf{{{m['sensor_angle']:.3f}°}}"
        ))
        st.caption(
            f"Con esta distancia, el sensor de reducción se activa en θ = {det_m['angle_sensor_red']:.2f}° "
            f"(t = {m['t_red']:.2f} s) y el de paro en θ = {det_m['angle_stop']:.2f}° (t = {m['t_stop']:.2f} s). "
            f"La rampa de DECEL necesita {det_m['angle_needed_decel']:.2f}° para llegar a SPEED_AUTO_SLOW."
        )

        st.divider()
        st.subheader("📏 Overrun angular y su conversión al riel de sensores (mm)")
        st.caption(
            "Convierte el overrun angular (causado por RAMP_STOP) a la distancia lineal equivalente sobre el "
            "riel de sensores, para que mantenimiento pueda reposicionar el sensor de paro usando cinta métrica."
        )
        overrun_mm_riel = angle_to_mm(m['ang_overrun'], R_sensor, modo_key)
        ov1, ov2, ov3 = st.columns(3)
        ov1.metric("Overrun Angular", f"{m['ang_overrun']:.4f}°")
        ov2.metric(f"Overrun en el Riel (R={R_sensor:.2f}mm)", f"{overrun_mm_riel:.3f} mm")
        ov3.metric("Overrun Time", f"{m['overrun_time']*1000:.0f} ms")

        st.markdown("**🔧 Recomendación de compensación (para preservar la posición de paro original):**")
        trigger_angle_original = ss.angle_total_mesa
        trigger_angle_compensado = trigger_angle_original - m['ang_overrun']
        trigger_mm_original = angle_to_mm(trigger_angle_original, R_sensor, modo_key)
        trigger_mm_compensado = angle_to_mm(trigger_angle_compensado, R_sensor, modo_key)

        comp1, comp2 = st.columns(2)
        comp1.metric("Posición Actual del Sensor de Paro", f"{trigger_mm_original:.2f} mm")
        comp2.metric("Posición Compensada Recomendada", f"{trigger_mm_compensado:.2f} mm",
                     delta=f"{(trigger_mm_compensado - trigger_mm_original):.3f} mm (mover hacia atrás)",
                     delta_color="inverse")
        st.info(
            f"💡 Para que la mesa termine deteniéndose exactamente en el mismo punto que tendría con "
            f"`RAMP_STOP MESA = 0`, mueve el **sensor de paro** físicamente "
            f"**{abs(trigger_mm_compensado - trigger_mm_original):.3f} mm hacia atrás** sobre el riel."
        )

    # ------------------------------------------
    with tab_eq:
        st.subheader("📐 Equivalencia lineal en el punto crítico (R_max)")
        st.caption(
            f"Convierte cada parámetro angular de la mesa a su equivalente lineal (mm/s, mm/s²) en el punto más "
            f"lejano de la pieza (R_max = {r_max_mesa:.1f} mm), para comparar directamente contra las mismas "
            f"unidades del conveyor lineal (Perfil A, configurado en la sección Conveyor)."
        )
        conv_a = simular_conveyor("a")
        st.plotly_chart(figura_equivalencia(m, conv_a))

        eqL1, eqL2 = st.columns(2)
        with eqL1:
            st.markdown("**🔄 Mesa Giratoria (convertido a R_max)**")
            st.metric("SPEED_AUTO_FAST MESA → mm/s", f"{det_m['v_fast_lineal']:.1f} mm/s")
            st.metric("SPEED_AUTO_SLOW MESA → mm/s", f"{det_m['v_slow_lineal']:.1f} mm/s")
            st.metric("RAMP_ACCEL MESA → mm/s²", f"{det_m['a_accel_lineal']:.1f} mm/s²")
            st.metric("RAMP_DECEL MESA → mm/s²", f"{det_m['a_decel_lineal']:.1f} mm/s²")
        with eqL2:
            st.markdown("**➡️ Conveyor Lineal (Perfil A, referencia directa)**")
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
# PÁGINA: CONVEYOR
# ==========================================
def sidebar_conveyor():
    sb = st.sidebar
    sb.header("➡️ Parámetros — Conveyor")

    sb.subheader("Geometría")
    sb.number_input("Largo Total Conveyor (mm)", step=100.0, key="conveyor_length")

    sb.subheader("🔵 Perfil A (principal)")
    sb.number_input("SPEED_AUTO_FAST A (mm/s)", step=10.0, key="speed_fast_a")
    sb.number_input("SPEED_AUTO_SLOW A (mm/s)", step=10.0, key="speed_slow_a")
    sb.number_input("RAMP_ACCEL A (mm/s²)", step=50.0, key="accel_a")
    sb.number_input("RAMP_DECEL A (mm/s²)", step=50.0, key="decel_a")
    sb.number_input("Distancia Sensor Reducción A (mm)", step=25.0, key="sensor_distance_a")
    sb.number_input(
        "RAMP_STOP A (mm/s²)", step=100.0, min_value=0.0, key="ramp_stop_a",
        help="Deceleración directamente comandada por el PLC para el frenado final — mismo tipo de parámetro que "
             "RAMP_ACCEL / RAMP_DECEL (NO es un tiempo). Un piso mecánico (backlash de cadena, flexión de chasis) "
             "limita la deceleración máxima físicamente alcanzable a v_slow / 20ms. Si se deja en 0, se asume que "
             "el PLC no limita el frenado y el sistema se detiene tan rápido como mecánicamente es posible."
    )
    sb.number_input("Coeficiente Fricción μ A", step=0.01, min_value=0.01, max_value=1.0, key="mu_a")

    sb.divider()
    sb.checkbox("Comparar con Perfil B", key="comparar")

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
        sb.subheader("🟣 Perfil B (comparativa)")
        sb.number_input("SPEED_AUTO_FAST B (mm/s)", step=10.0, key="speed_fast_b")
        sb.number_input("SPEED_AUTO_SLOW B (mm/s)", step=10.0, key="speed_slow_b")
        sb.number_input("RAMP_ACCEL B (mm/s²)", step=50.0, key="accel_b")
        sb.number_input("RAMP_DECEL B (mm/s²)", step=50.0, key="decel_b")
        sb.number_input("Distancia Sensor Reducción B (mm)", step=25.0, key="sensor_distance_b")
        sb.number_input(
            "RAMP_STOP B (mm/s²)", step=100.0, min_value=0.0, key="ramp_stop_b",
            help="Igual que RAMP_STOP A — deceleración directa (mm/s²), no un tiempo. 0 = usar el máximo mecánico."
        )
        sb.number_input("Coeficiente Fricción μ B", step=0.01, min_value=0.01, max_value=1.0, key="mu_b")


def figura_movimiento_conveyor(a, b=None):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42], vertical_spacing=0.08,
        subplot_titles=("Velocidad del conveyor (mm/s)",
                        "Aceleración sobre la pieza (G) — zona verde = dentro de μ"),
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
            x=s["t"][idx], y=s["vel"][idx], mode="lines", name=f"Velocidad {nombre}",
            line=dict(color=color, width=3, dash=dash),
            customdata=np.column_stack([s["pos"][idx], nombres[idx]]),
            hovertemplate=f"{nombre}: " + "%{y:.1f} mm/s · x = %{customdata[0]:.1f} mm · %{customdata[1]}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=s["t"][idx], y=g[idx], mode="lines", name=f"Aceleración {nombre}",
            line=dict(color=color, width=2, dash=dash, shape="vh"),
            hovertemplate=f"{nombre}: " + "%{y:.3f} G<extra></extra>",
        ), row=2, col=1)

    # Sensores
    for nombre, s, color, _ in perfiles:
        c_red = COL_SENSOR_RED if nombre == "A" else COL_B
        c_stop = COL_SENSOR_STOP if nombre == "A" else "#6f4a94"
        for fila in (1, 2):
            fig.add_vline(x=s["t_red"], line_width=2, line_dash="dot", line_color=c_red, row=fila, col=1)
            fig.add_vline(x=s["t_stop"], line_width=2, line_dash="dash", line_color=c_stop, row=fila, col=1)
        entrada_leyenda(fig, f"Sensor Reducción {nombre} · {s['t_red']:.2f}s · {s['det']['pos_sensor_red']:.0f} mm",
                        c_red, "dot", row=1, col=1)
        entrada_leyenda(fig, f"Sensor Paro {nombre} · {s['t_stop']:.2f}s · {s['det']['pos_stop']:.0f} mm",
                        c_stop, "dash", row=1, col=1)

    # Límite de fricción
    mu_a = a["mu"]
    fig.add_hrect(y0=-mu_a, y1=mu_a, fillcolor=COL_OK, opacity=0.08, line_width=0, layer="below", row=2, col=1)
    for signo in (1, -1):
        fig.add_hline(y=signo * mu_a, line_width=2, line_dash="dash", line_color=COL_SLIP, row=2, col=1)
    entrada_leyenda(fig, f"Límite de fricción ±μ{' A' if b is not None else ''} = {mu_a:.2f}",
                    COL_SLIP, "dash", row=2, col=1)
    mu_max = mu_a
    if b is not None and abs(b["mu"] - mu_a) > 1e-9:
        mu_max = max(mu_max, b["mu"])
        for signo in (1, -1):
            fig.add_hline(y=signo * b["mu"], line_width=2, line_dash="dot", line_color=COL_B, row=2, col=1)
        entrada_leyenda(fig, f"Límite de fricción ±μ B = {b['mu']:.2f}", COL_B, "dot", row=2, col=1)

    # Fases del perfil A
    det_a = a["det"]
    segs = segmentos_fase(a["t"], det_a["fase"])
    bandas_fase(fig, segs, lambda sg: abs(det_a["a_cmd"][sg["i1"]]) / 9810.0, mu_a, float(a["t"][-1]))

    g_top = max(g_abs_max, mu_max)
    fig.update_yaxes(title_text="mm/s", row=1, col=1)
    fig.update_yaxes(title_text="G", range=[-g_top * 1.25, g_top * 1.5], row=2, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
    leyenda_abajo(fig, 680)
    return fig


def tabla_fases_conveyor(s, perfil=None):
    det = s["det"]
    filas = []
    for sg in segmentos_fase(s["t"], det["fase"]):
        a, b = sg["i0"], sg["i1"]
        g = abs(det["a_cmd"][b]) / 9810.0
        fila = {"Perfil": perfil} if perfil else {}
        fila.update({
            "Fase": FASE_NOMBRE[sg["fase"]],
            "Estado PLC": sg["fase"],
            "Inicio (s)": round(sg["t0"], 3),
            "Duración (s)": round(sg["t1"] - sg["t0"], 3),
            "Recorrido (mm)": round(float(s["pos"][b] - s["pos"][a]), 2),
            "v inicio → fin (mm/s)": f"{s['vel'][a]:.0f} → {s['vel'][b]:.0f}",
            "Aceleración (G)": round(g, 3),
            "Estado": "🔴 Supera μ" if g > s["mu"] else "🟢 OK",
        })
        filas.append(fila)
    return filas


def figura_fases_conveyor(a, b=None):
    etiquetas = ["Arranque<br>ACCEL_FAST", "Frenado a creep<br>DECEL_TO_SLOW", "Frenado final<br>DECEL_TO_STOP"]

    def valores(s):
        return [s["det"]["g_accel"], s["det"]["g_decel"], s["g_conv"]]

    fig = go.Figure()
    ga = valores(a)
    if b is None:
        colores = [COL_SLIP if g > a["mu"] else COL_OK for g in ga]
    else:
        colores = COL_A
    fig.add_trace(go.Bar(x=etiquetas, y=ga, name="Perfil A" if b is not None else "Aceleración de la fase",
                         marker_color=colores,
                         text=[f"{g:.3f} G" for g in ga], textposition="outside", hovertemplate="%{y:.3f} G"))
    g_all = list(ga)
    if b is not None:
        gb = valores(b)
        g_all += gb
        fig.add_trace(go.Bar(x=etiquetas, y=gb, name="Perfil B", marker_color=COL_B,
                             text=[f"{g:.3f} G" for g in gb], textposition="outside", hovertemplate="%{y:.3f} G"))
    fig.add_hline(y=a["mu"], line_width=2, line_dash="dash", line_color=COL_SLIP)
    entrada_leyenda(fig, f"Límite de fricción μ{' A' if b is not None else ''} = {a['mu']:.2f}", COL_SLIP, "dash")
    mu_max = a["mu"]
    if b is not None and abs(b["mu"] - a["mu"]) > 1e-9:
        mu_max = max(mu_max, b["mu"])
        fig.add_hline(y=b["mu"], line_width=2, line_dash="dot", line_color=COL_B)
        entrada_leyenda(fig, f"Límite de fricción μ B = {b['mu']:.2f}", COL_B, "dot")
    fig.update_layout(
        title="Aceleración por transición de estados vs μ", barmode="group", height=400,
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

    st.title("➡️ Conveyor lineal")
    st.caption("Análisis en tiempo real del perfil de velocidad, posicionamiento y deslizamiento inercial de la pieza.")

    # ---- Alertas de configuración ----
    if det_a['insufficient_distance']:
        st.warning(
            f"⚠️ **Perfil A — Configuración inconsistente:** la rampa `RAMP_DECEL A` necesita "
            f"**{det_a['dist_needed_decel']:.1f} mm** para bajar de SPEED_AUTO_FAST a SPEED_AUTO_SLOW, "
            f"pero la `Distancia Sensor Reducción A` solo tiene **{ss.sensor_distance_a:.1f} mm**."
        )
    if comparar and det_b['insufficient_distance']:
        st.warning(
            f"⚠️ **Perfil B — Configuración inconsistente:** la rampa `RAMP_DECEL B` necesita "
            f"**{det_b['dist_needed_decel']:.1f} mm** para bajar de SPEED_AUTO_FAST a SPEED_AUTO_SLOW, "
            f"pero la `Distancia Sensor Reducción B` solo tiene **{ss.sensor_distance_b:.1f} mm**."
        )
    if det_a['mechanically_capped']:
        st.info(
            f"ℹ️ **Perfil A:** `RAMP_STOP A` configurado ({ss.ramp_stop_a:.1f} mm/s²) excede el "
            f"máximo mecánico alcanzable ({det_a['a_mechanical_max']:.1f} mm/s², limitado por el piso de "
            f"{T_MIN_MECANICO_MS:.0f} ms). Se aplicó el límite mecánico: deceleración real = "
            f"**{det_a['a_stop_conveyor']:.1f} mm/s²** (≈{det_a['ramp_stop_real_ms']:.1f} ms)."
        )
    if comparar and det_b['mechanically_capped']:
        st.info(
            f"ℹ️ **Perfil B:** `RAMP_STOP B` configurado ({ss.ramp_stop_b:.1f} mm/s²) excede el "
            f"máximo mecánico alcanzable ({det_b['a_mechanical_max']:.1f} mm/s²). Se aplicó el límite mecánico: "
            f"deceleración real = **{det_b['a_stop_conveyor']:.1f} mm/s²** (≈{det_b['ramp_stop_real_ms']:.1f} ms)."
        )

    # ---- Resumen ----
    def fila_resumen(s, etiqueta, ref=None):
        d = s["det"]
        pos_final = s["pos"][-1] + s["d_desliza"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"Estado {etiqueta} (fase más crítica)", estado_txt(d['se_desliza_alguna_fase']))
        c2.metric(f"Fase más crítica {etiqueta}", f"{d['g_max_fase']:.3f} G", help=d['peor_fase_label'],
                  delta=d['peor_fase_label'].split(" (")[0], delta_color="off")
        c3.metric(f"Factor de seguridad {etiqueta}", fmt_fs(d['factor_seguridad_fase']))
        if ref is None:
            c4.metric(f"Tiempo de ciclo {etiqueta}", f"{s['t'][-1]:.2f} s")
            c5.metric(f"Posición final pieza {etiqueta}", f"{pos_final:.2f} mm")
        else:
            c4.metric(f"Tiempo de ciclo {etiqueta}", f"{s['t'][-1]:.2f} s",
                      delta=f"{(s['t'][-1] - ref['t'][-1]):.2f} s vs A", delta_color="inverse")
            ref_pos = ref["pos"][-1] + ref["d_desliza"]
            c5.metric(f"Posición final pieza {etiqueta}", f"{pos_final:.2f} mm",
                      delta=f"{pos_final - ref_pos:.2f} mm vs A", delta_color="off")

    with st.container(border=True):
        st.markdown("**Resumen del ciclo**")
        fila_resumen(A, "A")
        if comparar:
            fila_resumen(B, "B", ref=A)

    tab_mov, tab_slip, tab_ab, tab_fund = st.tabs([
        "📈 Movimiento", "⚠️ Deslizamiento", "⚖️ Perfil A vs B", "📚 Fundamentos",
    ])

    # ------------------------------------------
    with tab_mov:
        st.plotly_chart(figura_movimiento_conveyor(A, B))
        st.caption(
            "Bandas: fases del PLC del Perfil A (gris = rampa, rojo = la aceleración supera μ). "
            "Pasa el cursor sobre la curva de velocidad para ver la posición x y la fase en cada instante."
        )

        st.subheader("🎯 Posicionamiento y tiempo de ciclo (Perfil A)")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Conveyor Overrun", f"{A['overrun']:.2f} mm")
        col6.metric("Overrun Time", f"{A['overrun_time']*1000:.0f} ms")
        col7.metric("Posición Final Pieza", f"{(A['pos'][-1] + A['d_desliza']):.2f} mm")
        col8.metric("Tiempo Total de Movimiento", f"{A['t'][-1]:.2f} s")

        st.subheader("🧭 Fases del ciclo")
        filas = tabla_fases_conveyor(A, "A" if comparar else None)
        if comparar:
            filas += tabla_fases_conveyor(B, "B")
        st.dataframe(filas, hide_index=True)

    # ------------------------------------------
    with tab_slip:
        st.plotly_chart(figura_fases_conveyor(A, B))

        st.subheader("Aceleración en Gs por transición de estados (Perfil A)")
        st.caption(
            "El conveyor lineal tiene tres transiciones de velocidad donde la pieza experimenta aceleración: "
            "el arranque (ACCEL_FAST), el frenado a velocidad de creep (DECEL_TO_SLOW), y el frenado final "
            "(DECEL_TO_STOP). Cada una se evalúa por separado contra μ."
        )
        fa1, fa2, fa3 = st.columns(3)
        fa1.metric("ACCEL_FAST (arranque)", f"{det_a['g_accel']:.3f} G",
                   delta="🔴 DESLIZA" if det_a['se_desliza_accel'] else "🟢 OK", delta_color="off")
        fa2.metric("DECEL_TO_SLOW (frenado a creep)", f"{det_a['g_decel']:.3f} G",
                   delta="🔴 DESLIZA" if det_a['se_desliza_decel'] else "🟢 OK", delta_color="off")
        fa3.metric("DECEL_TO_STOP (frenado final)", f"{A['g_conv']:.3f} G",
                   delta="🔴 DESLIZA" if det_a['se_desliza_stop_fase'] else "🟢 OK", delta_color="off")

        fb1, fb2 = st.columns(2)
        fb1.metric("🔺 Fase Más Crítica", det_a['peor_fase_label'].split(" (")[0], help=det_a['peor_fase_label'])
        fb2.metric("🛡️ Factor de Seguridad de la Fase Más Crítica", fmt_fs(det_a['factor_seguridad_fase']))

        if det_a['se_desliza_alguna_fase'] and not det_a['se_desliza_stop_fase']:
            st.warning(
                f"⚠️ **Diferencia detectada:** el frenado final está ESTABLE, pero **{det_a['peor_fase_label']}** "
                f"presenta {det_a['g_max_fase']:.3f} G, que **supera** μ ({ss.mu_a:.2f} G)."
            )
        elif det_a['se_desliza_alguna_fase']:
            st.error(f"🔴 Al menos una fase supera μ — la más crítica es **{det_a['peor_fase_label']}**.")
        else:
            st.success("🟢 Las tres transiciones de velocidad están dentro del límite de fricción.")

        st.divider()
        st.subheader("📊 Inercia y deslizamiento de la pieza en el frenado final (Perfil A)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Deceleración del Conveyor", f"{A['g_conv']:.3f} G")
        col2.metric("Límite de Fricción (μ)", f"{ss.mu_a:.2f} G")
        col3.metric("Estabilidad de la Carga", estado_txt(A['desliza']))
        col4.metric("Deslizamiento Relativo",
                    f"{A['d_desliza']:.3f} mm ({A['d_desliza']*1000:.0f} µm)" if A['desliza'] else "0.000 mm")

        col_s1, col_s2 = st.columns(2)
        col_s1.metric("🛡️ Factor de Seguridad (μ / g_conv)", fmt_fs(det_a['factor_seguridad']))
        col_s2.metric("🎯 μ mínimo requerido para NO deslizar", f"{det_a['mu_minimo_requerido']:.3f}")

        titulo = ("🔍 Ver desglose de cálculo paso a paso (Perfil A)" if A['desliza']
                  else "ℹ️ Ver cálculo de estabilidad y deceleración (Perfil A)")
        with st.expander(titulo):
            st.markdown("### 🧮 Desglose del cálculo (valores simulados)")

            st.markdown("**Paso 1: Deceleración mecánica máxima (piso)**")
            st.latex(L(
                f"a_{{~text{{mech,max}}}} = ~frac{{v_{{~text{{slow}}}}}}{{20~,~text{{ms}}}} = "
                f"~frac{{{ss.speed_slow_a:.1f}~,~text{{mm/s}}}}{{0.020~,~text{{s}}}} = "
                f"~mathbf{{{det_a['a_mechanical_max']:.1f}~,~text{{mm/s}}^2}}"
            ))

            st.markdown("**Paso 2: Deceleración efectiva de paro (RAMP_STOP limitado por el piso mecánico)**")
            st.latex(L(
                f"a_{{~text{{stop}}}} = ~min({ss.ramp_stop_a:.1f}~,~text{{mm/s}}^2, "
                f"{det_a['a_mechanical_max']:.1f}~,~text{{mm/s}}^2) = ~mathbf{{{det_a['a_stop_conveyor']:.1f}~,~text{{mm/s}}^2}} "
                f"~quad (t~approx{det_a['ramp_stop_real_ms']:.1f}~,~text{{ms}})"
            ))
            st.latex(L(
                f"g_{{~text{{conv}}}} = ~frac{{{det_a['a_stop_conveyor']:.1f}~,~text{{mm/s}}^2}}"
                f"{{9810~,~text{{mm/s}}^2}} = ~mathbf{{{A['g_conv']:.3f}~,~text{{G}}}}"
            ))

            st.markdown("**Paso 3: Aceleración máxima admisible por fricción**")
            st.latex(L(
                f"a_{{~text{{max~_piece}}}} = ~mu ~cdot g = {ss.mu_a:.2f} ~cdot 9810~,~text{{mm/s}}^2 "
                f"= {det_a['a_max_pieza']:.2f}~,~text{{mm/s}}^2 ~quad (~mu = ~mathbf{{{ss.mu_a:.2f}~,~text{{G}}}})"
            ))

            st.markdown("**Paso 4: Criterio de deslizamiento**")
            if A['desliza']:
                st.error(f"🔴 **DESLIZAMIENTO DETECTADO:** g_conv ({A['g_conv']:.3f} G) > μ ({ss.mu_a:.2f} G).")
                st.markdown("**Paso 5: Distancia de deslizamiento relativo (Δd)**")
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
                st.success(f"🟢 **CARGA ESTABLE:** g_conv ({A['g_conv']:.3f} G) ≤ μ ({ss.mu_a:.2f} G).")

    # ------------------------------------------
    with tab_ab:
        if not comparar:
            st.info("Activa **Comparar con Perfil B** en la barra lateral para comparar dos configuraciones del PLC. "
                    "El Perfil B arranca como copia del Perfil A.")
        else:
            st.subheader("Aceleración por transición de estados — Perfil B")
            st.caption("Los deltas son B − A (rojo = B exige más fricción que A).")
            fa1b, fa2b, fa3b = st.columns(3)
            fa1b.metric("ACCEL_FAST B", f"{det_b['g_accel']:.3f} G",
                        delta=f"{(det_b['g_accel'] - det_a['g_accel']):.3f} G", delta_color="inverse")
            fa2b.metric("DECEL_TO_SLOW B", f"{det_b['g_decel']:.3f} G",
                        delta=f"{(det_b['g_decel'] - det_a['g_decel']):.3f} G", delta_color="inverse")
            fa3b.metric("DECEL_TO_STOP B", f"{B['g_conv']:.3f} G",
                        delta=f"{(B['g_conv'] - A['g_conv']):.3f} G", delta_color="inverse")
            st.caption(f"Fase más crítica en B: **{det_b['peor_fase_label']}** ({det_b['g_max_fase']:.3f} G)")

            st.divider()
            st.subheader("⚖️ Perfil A vs Perfil B — frenado final y posicionamiento")
            c_b1, c_b2, c_b3, c_b4 = st.columns(4)
            c_b1.metric("Deceleración Perfil B", f"{B['g_conv']:.3f} G",
                        delta=f"{(B['g_conv'] - A['g_conv']):.3f} G", delta_color="inverse")
            c_b2.metric("Deslizamiento Pieza Perfil B", f"{B['d_desliza']:.3f} mm",
                        delta=f"{(B['d_desliza'] - A['d_desliza']):.3f} mm", delta_color="inverse")
            c_b3.metric("Overrun Perfil B", f"{B['overrun']:.2f} mm",
                        delta=f"{(B['overrun'] - A['overrun']):.2f} mm", delta_color="inverse")
            c_b4.metric("Overrun Time Perfil B", f"{B['overrun_time']*1000:.0f} ms",
                        delta=f"{(B['overrun_time'] - A['overrun_time'])*1000:.0f} ms", delta_color="inverse")

            c_b5, c_b6 = st.columns(2)
            c_b5.metric("Posición Final Pieza B", f"{(B['pos'][-1] + B['d_desliza']):.2f} mm",
                        delta=f"{(B['pos'][-1] + B['d_desliza']) - (A['pos'][-1] + A['d_desliza']):.2f} mm",
                        delta_color="inverse")
            c_b6.metric("Tiempo de Ciclo Perfil B", f"{B['t'][-1]:.2f} s",
                        delta=f"{(B['t'][-1] - A['t'][-1]):.2f} s", delta_color="inverse")

    # ------------------------------------------
    with tab_fund:
        fundamentos_conveyor()


# ==========================================
# FUNDAMENTOS MATEMÁTICOS Y FÍSICOS
# ==========================================
def fundamentos_comunes(unidad_a, simbolo):
    """Secciones compartidas: piso mecánico, deceleración efectiva y umbral de fricción."""
    st.subheader("1. Piso de elasticidad mecánica (deceleración máxima alcanzable)")
    st.markdown(L(
        f"`RAMP_STOP` es una **deceleración comandada directamente** ({unidad_a}), consistente con "
        "`RAMP_ACCEL`/`RAMP_DECEL` — NO es un tiempo. La elasticidad mecánica impone un tiempo mínimo de "
        "paro de $T_{~text{min}} = 20~,~text{ms}$, que se traduce en una deceleración máxima físicamente "
        "alcanzable:"
    ))
    if simbolo == "a":
        st.latex(L(r"a_{~text{mech,max}} = ~frac{v_{~text{slow}}}{T_{~text{min}}}"))
    else:
        st.latex(L(r"~alpha_{~text{mech,max}} = ~frac{~omega_{~text{slow}}}{T_{~text{min}}} ~quad [°/s^2]"))


def fundamentos_conveyor():
    st.header("📐 Motor físico y fórmulas cinemáticas — Conveyor")
    st.markdown(
        "Modelos analíticos usados para simular la dinámica del conveyor, las fuerzas de deceleración, "
        "los límites de fricción y el desplazamiento inercial de la pieza transportada."
    )
    fundamentos_comunes("mm/s²", "a")

    st.subheader("2. Deceleración efectiva de paro")
    st.latex(L(
        r"a_{~text{stop}} = ~begin{cases} a_{~text{mech,max}} & ~text{RAMP~_STOP} = 0 \\ "
        r"~min(~text{RAMP~_STOP}, a_{~text{mech,max}}) & ~text{RAMP~_STOP} > 0 ~end{cases}"
    ))
    st.latex(L(r"g_{~text{conv}} = ~frac{a_{~text{stop}}}{9810}"))

    st.subheader("3. Umbral de fricción estática y criterio de deslizamiento")
    st.latex(L(r"a_{~text{max~_piece}} = ~mu ~cdot g = ~mu ~cdot 9810~,~text{mm/s}^2"))

    st.subheader("4. Estimación del deslizamiento relativo de la pieza")
    st.latex(L(r"d_{~text{piece}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{max~_piece}}}"))
    st.latex(L(r"d_{~text{conveyor}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{stop}}}"))
    st.latex(L(r"~Delta d = d_{~text{piece}} - d_{~text{conveyor}}"))

    st.subheader("5. Factor de seguridad y fricción mínima requerida")
    st.latex(L(r"~text{Factor de Seguridad} = ~frac{~mu}{g_{~text{conv}}} ~qquad ~mu_{~text{min}} = g_{~text{conv}}"))

    st.subheader("6. Disposición de sensores: ambos cerca del final del recorrido")
    st.latex(L(r"P_{~text{stop}} = L_{~text{total}} ~quad P_{~text{reduction}} = L_{~text{total}} - S_{~text{distance}}"))

    st.subheader("7. Validación de la distancia entre sensores")
    st.latex(L(r"d_{~text{needed}} = ~frac{v_{~text{fast}}^2 - v_{~text{slow}}^2}{2 ~cdot ~text{RAMP~_DECEL}}"))

    st.subheader("8. Aceleración por transición de estados")
    st.latex(L(r"g_{~text{accel}} = ~frac{~text{RAMP~_ACCEL}}{9810}"))
    st.latex(L(r"g_{~text{decel}} = ~frac{~text{RAMP~_DECEL}}{9810}"))
    st.latex(L(r"g_{~text{stop}} = ~frac{a_{~text{stop}}}{9810}"))
    st.latex(L(r"g_{~text{max~_fase}} = ~max~left(g_{~text{accel}}, g_{~text{decel}}, g_{~text{stop}}~right)"))

    st.subheader("9. Máquina de estados e integración numérica")
    st.markdown(L(
        "El PLC se modela como la secuencia `ACCEL_FAST → CRUISE_FAST → DECEL_TO_SLOW → CRUISE_SLOW → "
        "DECEL_TO_STOP`, integrada con paso $~Delta t = 1~,~text{ms}$. La posición acumulada es la que "
        "dispara los sensores de reducción y paro:"
    ))
    st.latex(L(r"x_i = x_{i-1} + v_i ~cdot ~Delta t"))


def fundamentos_mesa():
    st.header("📐 Motor físico y fórmulas cinemáticas — Mesa giratoria")
    st.markdown(
        "Modelos analíticos usados para simular el giro de la mesa, las componentes tangencial y centrípeta "
        "que siente la pieza, y la conversión de posiciones angulares al riel de sensores."
    )
    fundamentos_comunes("°/s²", "alpha")

    st.subheader("2. Deceleración angular efectiva y componente tangencial (frenado)")
    st.latex(L(
        r"~alpha_{~text{stop}} = ~begin{cases} ~alpha_{~text{mech,max}} & ~text{RAMP~_STOP~_MESA} = 0 \\ "
        r"~min(~text{RAMP~_STOP~_MESA}, ~alpha_{~text{mech,max}}) & ~text{RAMP~_STOP~_MESA} > 0 ~end{cases}"
    ))
    st.latex(L(r"a_{~text{tan~_stop}} = ~alpha_{~text{stop,rad}} ~cdot R_{~text{max}} ~quad [~text{mm/s}^2]"))
    st.latex(L(r"g_{~text{conv}} = ~frac{a_{~text{tan~_stop}}}{9810}"))

    st.subheader("3. Umbral de fricción estática")
    st.latex(L(r"a_{~text{max~_piece}} = ~mu ~cdot g = ~mu ~cdot 9810~,~text{mm/s}^2"))

    st.subheader("4. Analogía angular → lineal")
    st.latex(L(r"R_{~text{max}} = ~frac{~text{Longitud Total de Pieza}}{2}"))
    st.latex(L(r"v_{~text{lineal}} = ~omega_{~text{rad}} ~cdot R_{~text{max}} ~quad a_{~text{lineal}} = ~alpha_{~text{rad}} ~cdot R_{~text{max}}"))

    st.subheader("5. Posición angular por integración en el tiempo")
    st.markdown(L(
        "El ángulo girado es la integral de la velocidad angular. En la simulación discreta es la suma "
        "acumulada que dispara los sensores de reducción y paro (el valor en cada instante aparece al pasar "
        "el cursor sobre la gráfica de velocidad):"
    ))
    st.latex(L(r"~theta(t) = ~int_0^t ~omega(~tau)~,d~tau ~qquad ~theta_i = ~theta_{i-1} + ~omega_i ~cdot ~Delta t"))

    st.subheader("6. Componente centrípeta (crucero a velocidad constante)")
    st.latex(L(r"a_{~text{cent}} = ~omega_{~text{rad}}^2 ~cdot R_{~text{max}} ~quad [~text{mm/s}^2]"))
    st.latex(L(r"g_{~text{cent}} = ~frac{a_{~text{cent}}}{9810}"))

    st.subheader("7. Aceleración resultante en los instantes críticos")
    st.latex(L(r"a_{~text{resultante}} = ~sqrt{a_{~text{tangencial}}^2 + a_{~text{centrípeta}}^2}"))

    st.subheader("8. Deslizamiento relativo al frenar")
    st.latex(L(r"d_{~text{piece}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{max~_piece}}} ~quad d_{~text{mesa}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{tan~_stop}}}"))
    st.latex(L(r"~Delta d = d_{~text{piece}} - d_{~text{mesa}} ~qquad ~Delta ~theta = ~frac{~Delta d}{R_{~text{max}}}"))

    st.subheader("9. Factor de seguridad")
    st.latex(L(r"~text{FS} = ~frac{~mu}{g} ~quad ~text{(frenado, crucero o resultante)}"))

    st.subheader("10. Sensores: posición y validación")
    st.latex(L(r"~theta_{~text{stop}} = ~theta_{~text{total}} ~quad ~theta_{~text{reduction}} = ~theta_{~text{total}} - ~theta_{~text{sensor~_dist}}"))
    st.latex(L(r"~theta_{~text{needed}} = ~frac{~omega_{~text{fast}}^2 - ~omega_{~text{slow}}^2}{2 ~cdot ~text{RAMP~_DECEL~_MESA}}"))

    st.subheader("11. Riel de sensores: conversión ángulo ↔ mm para mantenimiento")
    st.latex(L(r"~theta_{~text{sensor}} = ~frac{s_{~text{mm}}}{R_{~text{sensor}}} ~quad ~text{(Arco, radianes → grados)}"))
    st.latex(L(r"~theta_{~text{sensor}} = 2 ~arcsin~left(~frac{s_{~text{mm}}}{2R_{~text{sensor}}}~right) ~quad ~text{(Cuerda)}"))
    st.latex(L(r"s_{~text{arc}} = R_{~text{sensor}} ~cdot ~theta_{~text{rad}}"))
    st.latex(L(r"s_{~text{chord}} = 2R_{~text{sensor}} ~sin~left(~frac{~theta_{~text{rad}}}{2}~right)"))

    st.subheader("12. Limitación de alcance: deslizamiento libre en un marco giratorio")
    st.markdown(
        "Una simulación totalmente rigurosa de un objeto que desliza libremente sobre una plataforma giratoria "
        "requiere resolver el movimiento en un marco de referencia no inercial, con las pseudo-fuerzas de "
        "Coriolis y de Euler. Este modelo evalúa el umbral de deslizamiento, no la trayectoria posterior."
    )


# ==========================================
# NAVEGACIÓN: DOS SECCIONES
# ==========================================
pagina = st.navigation(
    [
        st.Page(pagina_mesa, title="Mesa giratoria", icon=":material/360:", url_path="mesa", default=True),
        st.Page(pagina_conveyor, title="Conveyor", icon=":material/conveyor_belt:", url_path="conveyor"),
    ],
    position="top",
)
pagina.run()
