import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
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
    page_icon=":material/settings:",
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

# Palette — industrial (RAL-inspired): anthracite ink, gentian blue, signal green/red for state only
FONT_UI = "Archivo, 'Helvetica Neue', Arial, sans-serif"
INK = "#1E2629"          # anthracite — text and the motion profile
INK_2 = "#4F5B61"        # secondary text
MUTED = "#7C878D"        # axis labels, captions
LINE = "#DCE1E4"         # hairlines
GRID = "#E8ECEE"         # chart gridlines
PANEL = "#FFFFFF"        # work surfaces
ACCENT = "#13568F"       # gentian blue — structure, Profile A
VIOLET = "#6B5BD6"       # Profile B / centripetal component
OK = "#237F52"           # signal green — within μ
BAD = "#B3261E"          # signal red — exceeds μ

# Colors (consistent across the app)
COL_A = ACCENT
COL_B = VIOLET
COL_B_STOP = "#4B3EA8"
COL_B_RED = "#A69DE8"
COL_SENSOR_RED = MUTED
COL_SENSOR_STOP = INK
COL_OK = OK
COL_SLIP = BAD
COL_TAN = "#9AA4AA"
COL_CENT = VIOLET
COL_RESULT = ACCENT
COL_VEL_MESA = INK
COL_NEUTRO = "#AEB8BD"

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
# VISUAL THEME (presentation only — no effect on any calculation)
# ==========================================
_EJE = dict(
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
    showline=True, linecolor="#C3CBD0", linewidth=1, ticks="outside", ticklen=4, tickwidth=1,
    tickcolor="#C3CBD0", tickfont=dict(size=11, color=MUTED), title=dict(font=dict(size=11, color=INK_2), standoff=10),
)
# Every trace sets its own color, so the template carries no color list. The theme is cosmetic:
# if this Plotly version rejects any of it, fall back to plain white instead of stopping the app.
try:
    pio.templates["ramps"] = go.layout.Template(layout=dict(
        font=dict(family=FONT_UI, size=12, color=INK),
        title=dict(font=dict(family=FONT_UI, size=14, color=INK), x=0.01, xanchor="left"),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        xaxis=_EJE, yaxis=_EJE,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=INK_2)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(family=FONT_UI, size=12, color=INK)),
        bargap=0.42, bargroupgap=0.12,
    ))
    pio.templates.default = "plotly_white+ramps"
except Exception:
    pio.templates.default = "plotly_white"

# Colors below mirror the palette constants above.
CSS_TEMA = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap');

.stApp {
  --ink: #1E2629; --ink-2: #4F5B61; --muted: #7C878D; --line: #DCE1E4; --line-2: #C3CBD0;
  --page: #F2F4F5; --panel: #FFFFFF; --track: #E9EDEF;
  --accent: #13568F; --violet: #6B5BD6; --ok: #237F52; --ok-wash: #E7F3EC; --bad: #B3261E; --bad-wash: #FBEAE8;
  --ui: Archivo, 'Helvetica Neue', Arial, sans-serif;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp summary { font-family: var(--ui); }

/* ---- Type scale: semi-condensed Archivo for headings and figures ---- */
.stApp h1, .stApp h2, .stApp h3, .stApp [data-testid="stMetricValue"], .rk-headline, .rk-figs dd {
  font-stretch: 87.5%;
}
.stApp [data-testid="stMain"] h1 { font-size: 2.15rem; font-weight: 700; letter-spacing: -.012em; color: var(--ink); }
.stApp h2 { font-size: 1.4rem; font-weight: 650; letter-spacing: -.005em; }
.stApp h3 { font-size: 1.15rem; font-weight: 650; color: var(--ink); }
.stApp [data-testid="stMain"] [data-testid="stHeading"]:has(h3) { margin-top: .6rem; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {
  color: var(--ink-2); font-size: .86rem; line-height: 1.55; max-width: 80ch;
}
.stApp [data-testid="stMarkdownContainer"] code {
  color: var(--ink); background: var(--track); border-radius: 4px; padding: .08em .35em; font-size: .85em;
}

/* ---- Chrome ---- */
.stApp header[data-testid="stHeader"] { background: var(--panel); border-bottom: 1px solid var(--line); }
.stApp [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.stApp [data-testid="stSidebar"] h2 { font-size: 1.2rem; }
.stApp [data-testid="stSidebar"] [data-testid="stHeading"]:has(h3) {
  margin-top: .9rem; padding-top: .9rem; border-top: 1px solid var(--line);
}
.stApp [data-testid="stSidebar"] h3 { font-size: .95rem; font-weight: 650; color: var(--ink); }
.stApp [data-testid="stWidgetLabel"] p { font-size: .84rem; color: var(--ink-2); }
.stApp [data-testid="stNumberInputField"] { font-variant-numeric: tabular-nums; }

/* ---- Metrics as quiet readouts ---- */
.stApp [data-testid="stMetric"] { border-left: 2px solid var(--line); padding: .15rem 0 .15rem .85rem; }
.stApp [data-testid="stMetricLabel"] p { font-size: .82rem; color: var(--ink-2); }
.stApp [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; color: var(--ink); letter-spacing: -.01em; }
.stApp [data-testid="stMetricDelta"] { font-size: .8rem; }

/* ---- Tabs ---- */
.stApp [data-testid="stTab"] p { font-size: .93rem; font-weight: 500; color: var(--muted); white-space: nowrap; }
.stApp [data-testid="stTab"][aria-selected="true"] p { color: var(--ink); font-weight: 650; }
.stApp [data-testid="stTab"]:hover p { color: var(--ink); }

/* ---- Surfaces: charts, expanders, alerts ---- */
.stApp [data-testid="stPlotlyChart"] {
  background: var(--panel); border-radius: 10px; box-shadow: 0 0 0 1px var(--line); overflow: hidden;
}
.stApp [data-testid="stExpander"] details { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
.stApp [data-testid="stExpander"] summary p { font-weight: 600; }
.stApp [data-testid="stAlertContainer"] { border-radius: 8px; }
.stApp [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p { font-size: .9rem; }

/* ---- Verdict panel (top of each section) ---- */
.rk-verdict { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.rk-row {
  display: grid; grid-template-columns: minmax(0, 1fr) 27rem; gap: 1.25rem 2.5rem; align-items: center;
  padding: 1.25rem 1.5rem 1.3rem 1.35rem; border-left: 6px solid var(--ok);
}
.rk-row + .rk-row { border-top: 1px solid var(--line); }
.rk-row.is-bad { border-left-color: var(--bad); }
.rk-profile { display: flex; align-items: center; gap: .5rem; font-size: .84rem; color: var(--ink-2); margin-bottom: .2rem; }
.rk-profile i { display: inline-block; width: 18px; height: 3px; border-radius: 2px; background: var(--key); }
.rk-headline { display: flex; align-items: center; gap: .6rem; font-size: 1.55rem; font-weight: 650; color: var(--ink); line-height: 1.2; }
.rk-icon {
  flex: none; display: inline-grid; place-items: center; width: 1.45rem; height: 1.45rem; border-radius: 50%;
  background: var(--ok); color: #fff; font-size: .85rem; font-weight: 700; font-stretch: 100%;
}
.is-bad .rk-icon { background: var(--bad); }
.rk-where { margin-top: .25rem; font-size: .9rem; color: var(--ink-2); }
.rk-meter { position: relative; margin-top: 1.4rem; max-width: 640px; }
.rk-track { position: relative; height: 10px; border-radius: 5px; background: var(--track); }
.rk-fill { position: absolute; inset: 0 auto 0 0; border-radius: 5px; background: var(--ok); }
.is-bad .rk-fill { background: var(--bad); }
.rk-limit { position: absolute; top: -6px; bottom: -6px; width: 2px; margin-left: -1px; background: var(--ink); }
.rk-limit span { position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); font-size: .78rem; color: var(--ink); line-height: 1.4; }
.rk-read { margin-top: .5rem; font-size: .86rem; color: var(--ink-2); }
.rk-read b { color: var(--ink); font-weight: 600; }
.rk-figs { display: grid; grid-template-columns: 1fr 1fr 1.35fr; gap: 0 1.5rem; margin: 0; }
.rk-figs div { min-width: 0; }
.rk-figs dt { font-size: .82rem; color: var(--ink-2); white-space: nowrap; }
.rk-figs dd { margin: .1rem 0 0; font-size: 1.75rem; font-weight: 600; color: var(--ink); line-height: 1.15; white-space: nowrap; }
.rk-figs dd small { display: block; font-size: .78rem; font-weight: 500; color: var(--ink-2); font-stretch: 100%; margin-top: .15rem; }
@media (max-width: 900px) {
  .rk-row { grid-template-columns: 1fr; padding: 1.1rem 1.1rem 1.2rem 1rem; }
  .rk-figs { display: flex; flex-wrap: wrap; gap: 1rem 1.75rem; }
  .rk-headline { font-size: 1.3rem; }
  .rk-figs dd { font-size: 1.45rem; }
}
"""


def aplicar_tema():
    """Inject the global CSS (style-only HTML takes no space in the layout)."""
    st.html(f"<style>{CSS_TEMA}</style>")


def panel_veredicto(filas):
    """Verdict panel: does the part slip, peak G against μ on a shared scale, and the key figures.

    filas: list of dicts with keys slip, peak, mu, where, figs [(label, value, note)], and optionally
    perfil / color for the A-vs-B comparison.
    """
    escala = max(max(f["peak"], f["mu"]) for f in filas) / 0.85
    escala = escala if escala > 0 else 1.0
    bloques = []
    for f in filas:
        estado = "is-bad" if f["slip"] else "is-ok"
        titular = "The part slips" if f["slip"] else "The part stays in place"
        icono = "✕" if f["slip"] else "✓"
        perfil = (f"<div class='rk-profile'><i style='--key:{f['color']}'></i>{f['perfil']}</div>"
                  if f.get("perfil") else "")
        relleno = min(f["peak"] / escala * 100, 100)
        limite = min(f["mu"] / escala * 100, 100)
        figs = "".join(
            f"<div><dt>{lab}</dt><dd>{val}{f'<small>{nota}</small>' if nota else ''}</dd></div>"
            for lab, val, nota in f["figs"]
        )
        bloques.append(
            f"<div class='rk-row {estado}'><div>{perfil}"
            f"<div class='rk-headline'><span class='rk-icon'>{icono}</span>{titular}</div>"
            f"<div class='rk-where'>Most critical point: {f['where']}</div>"
            f"<div class='rk-meter'><div class='rk-track'><div class='rk-fill' style='width:{relleno:.1f}%'></div>"
            f"<div class='rk-limit' style='left:{limite:.1f}%'><span>μ</span></div></div></div>"
            f"<div class='rk-read'>Peak <b>{f['peak']:.3f} G</b> against a friction limit of <b>{f['mu']:.2f} G</b></div>"
            f"</div><dl class='rk-figs'>{figs}</dl></div>"
        )
    st.html(f"<div class='rk-verdict'>{''.join(bloques)}</div>")


def aviso(tipo, texto):
    """Status message with its icon, so state never relies on color alone."""
    iconos = {"error": ":material/error:", "warning": ":material/warning:",
              "info": ":material/info:", "success": ":material/check_circle:"}
    getattr(st, tipo)(texto, icon=iconos[tipo])


def subtitulos_izquierda(fig):
    """Subplot titles as left-aligned chart headings (call right after make_subplots)."""
    fig.update_annotations(font=dict(family=FONT_UI, size=13, color=INK), x=0, xanchor="left")


aplicar_tema()

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
    return "∞" if fs == float('inf') else f"{fs:.2f}×"


def estado_txt(desliza):
    return "Slipping" if desliza else "Stable"


def delta_mu(g, mu):
    """Margin against μ for st.metric (use delta_color='inverse': red above μ, green below)."""
    return f"{g - mu:+.3f} G vs μ"  # keep the ASCII sign: Streamlit reads it for the arrow/color


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
            fig.add_vrect(x0=sg["t0"], x1=sg["t1"], fillcolor=COL_SLIP, opacity=0.09,
                          line_width=0, layer="below", row="all", col=1)
        elif sg["fase"] not in ("CRUISE_FAST", "CRUISE_SLOW"):
            fig.add_vrect(x0=sg["t0"], x1=sg["t1"], fillcolor=COL_NEUTRO, opacity=0.14,
                          line_width=0, layer="below", row="all", col=1)
        if t_total > 0 and (sg["t1"] - sg["t0"]) / t_total >= 0.07:
            fig.add_annotation(
                x=(sg["t0"] + sg["t1"]) / 2, y=0.98, xref=xref, yref=yref,
                text=FASE_CORTO[sg["fase"]], showarrow=False, yanchor="top",
                font=dict(size=10, color=COL_SLIP if desliza else MUTED),
            )


COL_OK_TXT = INK
COL_BORDE_OK = "#C3CBD0"


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


def texto_etiqueta(titulo, valor, desliza, detalle, color_titulo=INK_2):
    color_val = COL_SLIP if desliza else COL_OK_TXT
    return (f"<span style='font-size:10px;color:{color_titulo}'>{titulo}</span><br>"
            f"<span style='font-size:14px;color:{color_val}'><b>{valor}</b></span><br>"
            f"<span style='font-size:10px;color:{MUTED}'>{detalle}</span>")


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
            arrowhead=2, arrowsize=0.9, arrowwidth=1, arrowcolor=borde, standoff=3,
            bgcolor="rgba(255,255,255,0.97)", bordercolor=borde,
            borderwidth=2 if it.get("peor") else 1, borderpad=5,
            font=dict(family=FONT_UI, size=11, color=INK),
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
                             line=dict(color=color, width=1.5, dash=dash)), **pos)


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
    sb.header("Turntable parameters")

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
                        f"Acceleration on the part at R_max = {m['r_max']:.0f} mm (G) — green band = within μ"),
    )
    subtitulos_izquierda(fig)

    fig.add_trace(go.Scatter(
        x=t[idx], y=omega[idx], mode="lines", name="Angular velocity ω",
        line=dict(color=COL_VEL_MESA, width=2),
        customdata=np.column_stack([theta[idx], nombres[idx]]),
        hovertemplate="ω = %{y:.2f} °/s · θ = %{customdata[0]:.2f}° · %{customdata[1]}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t[idx], y=g_tan[idx], mode="lines", name="Tangential (ramp)",
        line=dict(color=COL_TAN, width=1.5, shape="vh"), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t[idx], y=g_cent[idx], mode="lines", name="Centripetal (ω²·R)",
        line=dict(color=COL_CENT, width=1.5, dash="dash"), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t[idx], y=g_res[idx], mode="lines", name="Resultant",
        line=dict(color=COL_RESULT, width=2), hovertemplate="%{y:.3f} G",
    ), row=2, col=1)

    mu = m["mu"]
    g_top = max(float(np.max(g_res)) if len(g_res) else 0.0, mu, 1e-3)
    fig.add_hrect(y0=0, y1=mu, fillcolor=COL_OK, opacity=0.07, line_width=0, layer="below", row=2, col=1)
    fig.add_hline(y=mu, line_width=1.5, line_dash="dash", line_color=COL_SLIP, row=2, col=1)
    entrada_leyenda(fig, f"Friction limit μ = {mu:.2f}", COL_SLIP, "dash", row=2, col=1)

    for fila in (1, 2):
        fig.add_vline(x=m["t_red"], line_width=1.2, line_dash="dot", line_color=COL_SENSOR_RED, row=fila, col=1)
        fig.add_vline(x=m["t_stop"], line_width=1.2, line_dash="dash", line_color=COL_SENSOR_STOP, row=fila, col=1)
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
            "Status": "✕ Exceeds μ" if res > m["mu"] else "✓ Within μ",
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
    fig.add_hline(y=mu, line_width=1.5, line_dash="dash", line_color=COL_SLIP)
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
                             line=dict(color=COL_A, width=2), hovertemplate="%{y:.1f} mm/s"))
    fig.add_trace(go.Scatter(x=conv_a["t"][idx_c], y=conv_a["vel"][idx_c], mode="lines",
                             name="Conveyor — Profile A", line=dict(color=COL_NEUTRO, width=1.8, dash="dash"),
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

    st.title("Turntable", anchor=False)
    st.caption(
        f"Part centered on the rotation axis. R_max (slip) = Length/2 = {r_max_mesa:.1f} mm. "
        f"Different from the Sensor Rail Radius ({R_sensor:.2f} mm), which is only used "
        f"to convert sensor positions between degrees and mm."
    )

    # ---- Verdict: combined worst case (tangential + centripetal) against μ ----
    panel_veredicto([dict(
        slip=det_m['se_desliza_resultante'], peak=det_m['g_res_max'], mu=ss.mu_mesa,
        where=det_m['peor_caso_label'],
        figs=[("Safety factor", fmt_fs(det_m['factor_seguridad_resultante']), "μ / peak"),
              ("Cycle time", f"{t_m[-1]:.2f} s", ""),
              ("Angular overrun", f"{m['ang_overrun']:.3f}°", "")],
    )])

    # ---- Configuration alerts ----
    if det_m['insufficient_angle']:
        aviso("warning",
            f"**Inconsistent configuration:** `RAMP_DECEL TABLE` needs "
            f"**{det_m['angle_needed_decel']:.1f}°** to go from SPEED_AUTO_FAST down to SPEED_AUTO_SLOW, "
            f"but the `Slowdown Sensor Distance TABLE` ({ss.sensor_distance_mm_mesa:.1f} mm ≈ "
            f"{m['sensor_angle']:.1f}°) is not enough."
        )
    if det_m['mechanically_capped']:
        aviso("info",
            f"Configured `RAMP_STOP TABLE` ({ss.ramp_stop_mesa:.1f} °/s²) exceeds the "
            f"achievable mechanical maximum ({det_m['alpha_mechanical_max']:.1f} °/s², limited by the "
            f"{T_MIN_MECANICO_MS:.0f} ms floor). The mechanical limit was applied: actual angular deceleration = "
            f"**{det_m['alpha_stop']:.1f} °/s²** (≈{det_m['ramp_stop_real_ms']:.1f} ms)."
        )

    tab_mov, tab_slip, tab_sens, tab_eq, tab_fund = st.tabs([
        "Motion", "Slip", "Sensors & adjustment", "Conveyor equivalence",
        "Fundamentals",
    ])

    # ------------------------------------------
    with tab_mov:
        st.plotly_chart(figura_movimiento_mesa(m), theme=None)
        st.caption(
            "Labels: maximum resultant acceleration of each phase with its tangential (ramp) and "
            "centripetal (ω²·R) components — dark = within μ, red = exceeds μ, thick border = most critical phase. "
            "Bands: PLC phases. Hover over the velocity curve to see the angle θ at each instant."
        )

        st.subheader("Positioning & cycle time")
        mp1, mp2, mp3, mp4, mp5 = st.columns(5)
        mp1.metric("Angular Overrun", f"{m['ang_overrun']:.3f}°")
        mp2.metric("Overrun Time", f"{m['overrun_time']*1000:.0f} ms")
        mp3.metric("Final Table Angle", f"{theta_m[-1]:.3f}°",
                   help=f"Should be the Total Rotation Angle ({ss.angle_total_mesa:.1f}°) plus the overrun.")
        mp4.metric("Final Part Position", f"{(theta_m[-1] + det_m['deslizamiento_deg_stop']):.3f}°")
        mp5.metric("Total Cycle Time", f"{t_m[-1]:.2f} s")

        st.subheader("Cycle phases")
        st.dataframe(tabla_fases_mesa(m), hide_index=True)

    # ------------------------------------------
    with tab_slip:
        st.plotly_chart(figura_instantes_mesa(m), theme=None)

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
                       delta="Worst case" if is_worst else None, delta_color="off", help=label)

        rc1, rc2 = st.columns(2)
        rc1.metric("Combined Worst Case (Max Resultant)", f"{det_m['g_res_max']:.3f} G",
                   help=f"Occurs at: {det_m['peor_caso_label']}")
        rc2.metric("Resultant Safety Factor (μ / g_res_max)", fmt_fs(det_m['factor_seguridad_resultante']))

        if det_m['se_desliza_resultante'] and not m['desliza']:
            aviso("warning",
                f"**Difference detected:** the current criterion (tangential only) says **STABLE**, but the "
                f"resultant criterion shows that at **{det_m['peor_caso_label']}** the total force "
                f"({det_m['g_res_max']:.3f} G) **exceeds** μ ({ss.mu_mesa:.2f} G)."
            )
        elif det_m['se_desliza_resultante']:
            aviso("error", f"The resultant criterion confirms slip — worst case at **{det_m['peor_caso_label']}**.")
        else:
            aviso("success", "Even considering the combined worst case, the part stays stable throughout the cycle.")

        st.divider()
        st.subheader("Final stop (tangential component)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tangential Deceleration (stop)", f"{m['g_conv']:.3f} G")
        m2.metric("Friction Limit (μ)", f"{ss.mu_mesa:.2f} G")
        m3.metric("Stability When Stopping", estado_txt(m['desliza']),
                  delta=delta_mu(m['g_conv'], ss.mu_mesa), delta_color="inverse")
        m4.metric("Relative Slip",
                  f"{det_m['deslizamiento_deg_stop']:.4f}° ({m['d_desliza_mm']:.3f} mm)" if m['desliza'] else "0.0000°")
        st.metric("Safety Factor When Stopping (μ / g_tan)", fmt_fs(det_m['factor_seguridad_stop']))

        st.divider()
        st.subheader("Centripetal risk while cruising (constant speed)")
        st.caption(
            "This risk exists ANY time the table rotates at constant speed, even without braking. "
            "It has no equivalent on the linear conveyor."
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Centripetal G @ Fast Speed", f"{det_m['g_cent_fast']:.3f} G")
        mc2.metric("Stability @ Fast", "Risk" if det_m['se_desliza_cruise_fast'] else "OK",
                   delta=delta_mu(det_m['g_cent_fast'], ss.mu_mesa), delta_color="inverse")
        mc3.metric("Centripetal G @ Slow Speed", f"{det_m['g_cent_slow']:.3f} G")
        mc4.metric("Stability @ Slow", "Risk" if det_m['se_desliza_cruise_slow'] else "OK",
                   delta=delta_mu(det_m['g_cent_slow'], ss.mu_mesa), delta_color="inverse")

        if det_m['se_desliza_cruise_fast'] or det_m['se_desliza_cruise_slow']:
            aviso("error",
                "**Slip risk while cruising detected.** The centripetal force at this angular speed "
                "and radius exceeds the available friction."
            )
        else:
            aviso("success", "The centripetal force at both cruise speeds is within the friction limit.")

        with st.expander("View calculation breakdown — Turntable"):
            st.markdown("### Calculation Report (Simulated Values)")

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
                aviso("error", f"**SLIPS WHEN STOPPING:** g_conv ({m['g_conv']:.3f} G) > μ ({ss.mu_mesa:.2f} G)")
            else:
                aviso("success", f"**STABLE WHEN STOPPING:** g_conv ({m['g_conv']:.3f} G) ≤ μ ({ss.mu_mesa:.2f} G)")

    # ------------------------------------------
    with tab_sens:
        st.subheader("Distance between sensors (mm) — field setup")
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
        st.subheader("Angular overrun and its conversion to the sensor rail (mm)")
        st.caption(
            "Converts the angular overrun (caused by RAMP_STOP) into the equivalent linear distance on the "
            "sensor rail, so maintenance can reposition the stop sensor with a tape measure."
        )
        overrun_mm_riel = angle_to_mm(m['ang_overrun'], R_sensor, modo_key)
        ov1, ov2, ov3 = st.columns(3)
        ov1.metric("Angular Overrun", f"{m['ang_overrun']:.4f}°")
        ov2.metric(f"Overrun on the Rail (R={R_sensor:.2f}mm)", f"{overrun_mm_riel:.3f} mm")
        ov3.metric("Overrun Time", f"{m['overrun_time']*1000:.0f} ms")

        st.markdown("**Compensation recommendation (to keep the original stop position):**")
        trigger_angle_original = ss.angle_total_mesa
        trigger_angle_compensado = trigger_angle_original - m['ang_overrun']
        trigger_mm_original = angle_to_mm(trigger_angle_original, R_sensor, modo_key)
        trigger_mm_compensado = angle_to_mm(trigger_angle_compensado, R_sensor, modo_key)

        comp1, comp2 = st.columns(2)
        comp1.metric("Current Stop Sensor Position", f"{trigger_mm_original:.2f} mm")
        comp2.metric("Recommended Compensated Position", f"{trigger_mm_compensado:.2f} mm",
                     delta=f"{(trigger_mm_compensado - trigger_mm_original):.3f} mm (move back)",
                     delta_color="inverse")
        aviso("info",
            f"For the table to stop at exactly the same point it would reach with "
            f"`RAMP_STOP TABLE = 0`, physically move the **stop sensor** "
            f"**{abs(trigger_mm_compensado - trigger_mm_original):.3f} mm back** along the rail."
        )

    # ------------------------------------------
    with tab_eq:
        st.subheader("Linear equivalence at the critical point (R_max)")
        st.caption(
            f"Converts each angular table parameter into its linear equivalent (mm/s, mm/s²) at the farthest "
            f"point of the part (R_max = {r_max_mesa:.1f} mm), to compare directly against the same "
            f"units on the linear conveyor (Profile A, set up in the Conveyor section)."
        )
        conv_a = simular_conveyor("a")
        st.plotly_chart(figura_equivalencia(m, conv_a), theme=None)

        eqL1, eqL2 = st.columns(2)
        with eqL1:
            st.markdown("**Turntable (converted to R_max)**")
            st.metric("SPEED_AUTO_FAST TABLE → mm/s", f"{det_m['v_fast_lineal']:.1f} mm/s")
            st.metric("SPEED_AUTO_SLOW TABLE → mm/s", f"{det_m['v_slow_lineal']:.1f} mm/s")
            st.metric("RAMP_ACCEL TABLE → mm/s²", f"{det_m['a_accel_lineal']:.1f} mm/s²")
            st.metric("RAMP_DECEL TABLE → mm/s²", f"{det_m['a_decel_lineal']:.1f} mm/s²")
        with eqL2:
            st.markdown("**Linear Conveyor (Profile A, direct reference)**")
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
    sb.header("Conveyor parameters")

    sb.subheader("Geometry")
    sb.number_input("Total Conveyor Length (mm)", step=100.0, key="conveyor_length")

    sb.subheader("Profile A (main)")
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
        sb.subheader("Profile B (comparison)")
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
                        "Acceleration on the part (G) — green band = within μ"),
    )
    subtitulos_izquierda(fig)
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
            line=dict(color=color, width=2, dash=dash),
            customdata=np.column_stack([s["pos"][idx], nombres[idx]]),
            hovertemplate=f"{nombre}: " + "%{y:.1f} mm/s · x = %{customdata[0]:.1f} mm · %{customdata[1]}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=s["t"][idx], y=g[idx], mode="lines", name=f"Acceleration {nombre}",
            line=dict(color=color, width=1.6, dash=dash, shape="vh"),
            hovertemplate=f"{nombre}: " + "%{y:.3f} G<extra></extra>",
        ), row=2, col=1)

    # Sensors
    for nombre, s, color, _ in perfiles:
        c_red = COL_SENSOR_RED if nombre == "A" else COL_B_RED
        c_stop = COL_SENSOR_STOP if nombre == "A" else COL_B_STOP
        for fila in (1, 2):
            fig.add_vline(x=s["t_red"], line_width=1.2, line_dash="dot", line_color=c_red, row=fila, col=1)
            fig.add_vline(x=s["t_stop"], line_width=1.2, line_dash="dash", line_color=c_stop, row=fila, col=1)
        entrada_leyenda(fig, f"Slowdown Sensor {nombre} · {s['t_red']:.2f}s · {s['det']['pos_sensor_red']:.0f} mm",
                        c_red, "dot", row=1, col=1)
        entrada_leyenda(fig, f"Stop Sensor {nombre} · {s['t_stop']:.2f}s · {s['det']['pos_stop']:.0f} mm",
                        c_stop, "dash", row=1, col=1)

    # Friction limit
    mu_a = a["mu"]
    fig.add_hrect(y0=-mu_a, y1=mu_a, fillcolor=COL_OK, opacity=0.07, line_width=0, layer="below", row=2, col=1)
    for signo in (1, -1):
        fig.add_hline(y=signo * mu_a, line_width=1.5, line_dash="dash", line_color=COL_SLIP, row=2, col=1)
    entrada_leyenda(fig, f"Friction limit ±μ{' A' if b is not None else ''} = {mu_a:.2f}",
                    COL_SLIP, "dash", row=2, col=1)
    mu_max = mu_a
    if b is not None and abs(b["mu"] - mu_a) > 1e-9:
        mu_max = max(mu_max, b["mu"])
        for signo in (1, -1):
            fig.add_hline(y=signo * b["mu"], line_width=1.5, line_dash="dot", line_color=COL_B, row=2, col=1)
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
                                        color_titulo=color if b is not None else INK_2),
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
            "Status": "✕ Exceeds μ" if g > s["mu"] else "✓ Within μ",
        })
        filas.append(fila)
    return filas


def zona_paro(s, tau_rango=None):
    """Conveyor and part position around the stop sensor, on a time axis relative to the trigger.

    The conveyor comes from the simulation. The part adds the slip that builds up while it brakes at
    μ·g instead of following the conveyor — the same model as the slip metrics, so its curve ends
    exactly at the reported final part position.
    """
    det = s["det"]
    if s["t_stop"] <= 0:
        return None
    a_s = det["a_stop_conveyor"]
    v0 = float(np.sqrt(max(2.0 * a_s * det["dist_freno_conveyor"], 0.0)))  # = SPEED_AUTO_SLOW
    a_p = det["a_max_pieza"] if s["desliza"] else a_s
    t_conv = max(det["t_fully_stopped"] - s["t_stop"], 0.0)
    t_part = v0 / a_p if (s["desliza"] and a_p > 0) else t_conv
    t_fin = max(t_conv, t_part, 0.005)
    # Shared time window when several profiles are drawn together (curves hold their rest value)
    lo, hi = tau_rango if tau_rango else (-0.6 * t_fin, 1.15 * t_fin)
    tau = np.linspace(lo, hi, 600)
    conv = np.interp(s["t_stop"] + tau, s["t"], s["pos"]) - det["pos_stop"]

    def recorrido(a, tt):
        tt = np.clip(tt, 0.0, v0 / a if a > 0 else 0.0)
        return v0 * tt - 0.5 * a * tt ** 2

    slip = (recorrido(a_p, tau) - recorrido(a_s, tau)) if s["desliza"] else np.zeros_like(tau)
    conv_fin = float(s["pos"][-1] - det["pos_stop"])
    return dict(tau_ms=tau * 1000.0, conv=conv, part=conv + slip, t_fin=t_fin, t_conv_ms=t_conv * 1000.0,
                t_part_ms=t_part * 1000.0, conv_fin=conv_fin, part_fin=conv_fin + s["d_desliza"],
                pos_stop=det["pos_stop"])


def figura_zona_paro(perfiles):
    """Where the conveyor and the part come to rest after the stop sensor triggers."""
    fig = go.Figure()
    finales = []   # (y, text, color) of each resting point
    y_lo, y_hi = 0.0, 0.0
    pos_stop = None
    previos = [zona_paro(s) for _, s, _, _ in perfiles]
    t_fin = max([z["t_fin"] for z in previos if z is not None] or [0.005])
    # Curves continue flat after everything has stopped, leaving room for the Δ dimension lines
    x_fin = 2.1 * t_fin * 1000.0
    cotas = []     # (x, Δ, profile, color) of each stop-offset dimension line
    for (nombre, s, color, dash), z0 in zip(perfiles, previos):
        if z0 is None:
            continue
        z = zona_paro(s, tau_rango=(-0.35 * t_fin, 2.1 * t_fin))
        pos_stop = z["pos_stop"]
        cotas.append(((1.25 + 0.55 * len(cotas)) * t_fin * 1000.0, z["part_fin"], nombre, color))
        y_lo = min(y_lo, float(np.min(z["conv"])))
        y_hi = max(y_hi, float(np.max(z["part"])))
        nombre_conv = f"{nombre} · conveyor" if s["desliza"] else f"{nombre} · conveyor = part (no slip)"
        fig.add_trace(go.Scatter(
            x=z["tau_ms"], y=z["conv"], mode="lines", name=nombre_conv,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"{nombre} conveyor: " + "%{y:+.2f} mm<extra></extra>",
        ))
        if s["desliza"]:
            fig.add_trace(go.Scatter(
                x=z["tau_ms"], y=z["part"], mode="lines", name=f"{nombre} · part (slides on the conveyor)",
                line=dict(color=color, width=2, dash="dot"),
                hovertemplate=f"{nombre} part: " + "%{y:+.2f} mm<extra></extra>",
            ))
            finales.append((z["conv_fin"], f"{nombre} conveyor · {pos_stop + z['conv_fin']:.2f} mm "
                                           f"({z['conv_fin']:+.2f})", color))
            finales.append((z["part_fin"], f"<b>{nombre} part · {pos_stop + z['part_fin']:.2f} mm</b> "
                                           f"({z['part_fin']:+.2f})", color))
        else:
            finales.append((z["part_fin"], f"<b>{nombre} conveyor = part · {pos_stop + z['part_fin']:.2f} mm</b> "
                                           f"({z['part_fin']:+.2f})", color))
    if pos_stop is None:
        return None

    altura, m_top, m_bot = 470, 60, 120
    # Focus on where things come to rest: the approach before the sensor may leave through the bottom
    y_top = max(y_hi, 1e-3)
    y_lo, y_hi = max(y_lo, -0.55 * y_top) - 0.08 * y_top, y_top * 1.22
    # Resting-point labels in a column to the right of the curves, spread so they never overlap
    sep = (y_hi - y_lo) * 20.0 / (altura - m_top - m_bot)
    orden = sorted(range(len(finales)), key=lambda i: finales[i][0])
    ys = [finales[i][0] for i in orden]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + sep)
    x_lab = x_fin * 1.03
    for i, y_lab in zip(orden, ys):
        y_pt, texto, color = finales[i]
        fig.add_trace(go.Scatter(x=[x_fin], y=[y_pt], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=8, color=color, line=dict(color=PANEL, width=2))))
        fig.add_annotation(x=x_fin, y=y_pt, ax=x_lab, ay=y_lab, xref="x", yref="y", axref="x", ayref="y",
                           text=texto, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=color,
                           xanchor="left", standoff=5, font=dict(family=FONT_UI, size=11, color=INK))
    y_hi = max(y_hi, ys[-1] + sep)

    # Δ stop offset: dimension line from the stop sensor to where the part rests, with the sensor correction
    caja = (y_hi - y_lo) * 42.0 / (altura - m_top - m_bot)   # label box height in data units
    # Box centered on its dimension line, or just above it when the line is shorter than the box
    y_cajas = [d / 2.0 if d > 1.3 * caja else d + 0.65 * caja for _, d, _, _ in cotas]
    for i in sorted(range(len(cotas)), key=lambda i: y_cajas[i])[1:]:
        otros = [y_cajas[j] for j in range(len(cotas)) if j != i and y_cajas[j] <= y_cajas[i]]
        if otros and y_cajas[i] - max(otros) < caja:
            y_cajas[i] = max(otros) + caja
    for (x_c, delta, nombre, color), y_caja in zip(cotas, y_cajas):
        fig.add_annotation(x=x_c, y=delta, ax=x_c, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           text="", showarrow=True, arrowside="end+start", arrowhead=2, arrowsize=0.9,
                           arrowwidth=1.4, arrowcolor=color)
        titulo = f"{nombre} Δ" if len(cotas) > 1 else "Δ"
        fig.add_annotation(
            x=x_c, y=y_caja, xref="x", yref="y", showarrow=False, align="center",
            text=(f"<b>{titulo} {f'{delta:+.2f}'.replace('-', '−')} mm</b><br>"
                  f"<span style='font-size:10px;color:{INK_2}'>sensor {f'{-delta:+.2f}'.replace('-', '−')}</span>"),
            bgcolor="rgba(255,255,255,0.96)", bordercolor=color, borderwidth=1, borderpad=4,
            font=dict(family=FONT_UI, size=11, color=INK),
        )

    fig.add_hline(y=0, line_width=1.2, line_dash="dash", line_color=COL_SENSOR_STOP)
    fig.add_vline(x=0, line_width=1.2, line_dash="dot", line_color=MUTED)
    entrada_leyenda(fig, f"Stop sensor position · {pos_stop:.0f} mm", COL_SENSOR_STOP, "dash")
    entrada_leyenda(fig, "Stop sensor triggers (t = 0)", MUTED, "dot")
    fig.update_layout(title="Stop zone — where the conveyor and the part come to rest")
    fig.update_xaxes(title_text="Time after the stop sensor triggers (ms)",
                     range=[float(-0.35 * t_fin * 1000.0), x_fin * 1.75])
    fig.update_yaxes(title_text="Distance past the stop sensor (mm)", range=[y_lo, y_hi])
    leyenda_abajo(fig, altura, legend_y=-0.2)
    fig.update_layout(margin=dict(t=m_top, b=m_bot))
    return fig


def correccion_sensores(s, sensor_distance):
    """How far to move the sensors so the part rests exactly where the stop sensor is today.

    The approach only depends on the sensors' positions relative to each other, so shifting both by −Δ
    shifts the whole stop by −Δ. Moving only the stop sensor also works while the creep stretch
    (distance covered at SPEED_AUTO_SLOW before the stop sensor) is longer than Δ.
    """
    z = zona_paro(s)
    if z is None:
        return None
    det = s["det"]
    delta = z["part_fin"]
    return dict(delta=delta, stop_nuevo=det["pos_stop"] - delta, red_nuevo=det["pos_sensor_red"] - delta,
                creep=sensor_distance - det["dist_needed_decel"])


def texto_zona_paro(s, nombre):
    z = zona_paro(s)
    if z is None:
        return f"**{nombre}:** the stop sensor is never reached in the simulated 30 s."
    final = z["pos_stop"] + z["part_fin"]
    if s["desliza"]:
        return (f"**{nombre}:** the conveyor stops {z['conv_fin']:.2f} mm past the sensor in {z['t_conv_ms']:.0f} ms. "
                f"That stop is {s['g_conv']:.3f} G, above μ = {s['mu']:.2f}, so the part keeps sliding "
                f"**{s['d_desliza']:.2f} mm** for {z['t_part_ms']:.0f} ms and rests at **{final:.2f} mm**.")
    return (f"**{nombre}:** conveyor and part stop together {z['conv_fin']:.2f} mm past the sensor in "
            f"{z['t_conv_ms']:.0f} ms ({s['g_conv']:.3f} G ≤ μ = {s['mu']:.2f}) and rest at **{final:.2f} mm**.")


def clave_perfil(nombre, color):
    """Small colored key + profile name, so each row says which profile it belongs to."""
    st.html(f"<div class='rk-profile'><i style='--key:{color}'></i>{nombre}</div>")


def tablas_comparacion(A, B, ss):
    """Every key value of both profiles side by side, with B − A."""
    def fila(nombre, va, vb, fmt, unidad="", delta=True):
        d = f"{vb - va:+{fmt}}{unidad}".replace("-", "−") if delta else "—"
        return {"": nombre, "Profile A": f"{va:{fmt}}{unidad}", "Profile B": f"{vb:{fmt}}{unidad}", "B − A": d}

    da, db = A["det"], B["det"]
    ajustes = [
        fila("SPEED_AUTO_FAST", ss.speed_fast_a, ss.speed_fast_b, ".1f", " mm/s"),
        fila("SPEED_AUTO_SLOW", ss.speed_slow_a, ss.speed_slow_b, ".1f", " mm/s"),
        fila("RAMP_ACCEL", ss.accel_a, ss.accel_b, ".0f", " mm/s²"),
        fila("RAMP_DECEL", ss.decel_a, ss.decel_b, ".0f", " mm/s²"),
        fila("Slowdown sensor distance", ss.sensor_distance_a, ss.sensor_distance_b, ".1f", " mm"),
        fila("RAMP_STOP (configured)", ss.ramp_stop_a, ss.ramp_stop_b, ".0f", " mm/s²"),
        fila("Stop deceleration actually applied", da["a_stop_conveyor"], db["a_stop_conveyor"], ".0f", " mm/s²"),
        fila("Friction coefficient μ", A["mu"], B["mu"], ".2f"),
    ]
    acel = [
        fila("ACCEL_FAST (start-up)", da["g_accel"], db["g_accel"], ".3f", " G"),
        fila("DECEL_TO_SLOW (braking to creep)", da["g_decel"], db["g_decel"], ".3f", " G"),
        fila("DECEL_TO_STOP (final stop)", A["g_conv"], B["g_conv"], ".3f", " G"),
        {"": "Most critical phase", "Profile A": da["peor_fase_label"].split(" (")[0],
         "Profile B": db["peor_fase_label"].split(" (")[0], "B − A": "—"},
        {"": "Safety factor (μ / peak)", "Profile A": fmt_fs(da["factor_seguridad_fase"]),
         "Profile B": fmt_fs(db["factor_seguridad_fase"]), "B − A": "—"},
        {"": "Part slips?", "Profile A": estado_txt(da["se_desliza_alguna_fase"]),
         "Profile B": estado_txt(db["se_desliza_alguna_fase"]), "B − A": "—"},
    ]
    paro = [
        fila("Stop sensor triggers at", A["t_stop"], B["t_stop"], ".3f", " s"),
        fila("Stop duration (overrun time)", A["overrun_time"] * 1000, B["overrun_time"] * 1000, ".0f", " ms"),
        fila("Conveyor overrun past the stop sensor", A["overrun"], B["overrun"], ".2f", " mm"),
        fila("Conveyor rests at", A["pos"][-1], B["pos"][-1], ".2f", " mm"),
        fila("Part slip on the conveyor", A["d_desliza"], B["d_desliza"], ".2f", " mm"),
        fila("Final part position", A["pos"][-1] + A["d_desliza"], B["pos"][-1] + B["d_desliza"], ".2f", " mm"),
        fila("Total motion time", A["t"][-1], B["t"][-1], ".2f", " s"),
    ]
    return ajustes, acel, paro


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
    fig.add_hline(y=a["mu"], line_width=1.5, line_dash="dash", line_color=COL_SLIP)
    entrada_leyenda(fig, f"Friction limit μ{' A' if b is not None else ''} = {a['mu']:.2f}", COL_SLIP, "dash")
    mu_max = a["mu"]
    if b is not None and abs(b["mu"] - a["mu"]) > 1e-9:
        mu_max = max(mu_max, b["mu"])
        fig.add_hline(y=b["mu"], line_width=1.5, line_dash="dot", line_color=COL_B)
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

    st.title("Linear conveyor", anchor=False)
    st.caption("Real-time analysis of the velocity profile, positioning and inertial slip of the part.")

    # ---- Verdict per profile: most critical phase against μ ----
    def fila_veredicto(s, perfil, color, ref=None):
        d = s["det"]
        pos_final = s["pos"][-1] + s["d_desliza"]
        nota_t = ""
        # Say when the final position includes slip, so it is not read as the conveyor stop
        notas_p = [f"incl. {s['d_desliza']:.2f} mm slip" if s["desliza"] else "no slip"]
        if ref is not None:
            nota_t = f"{s['t'][-1] - ref['t'][-1]:+.2f} s vs A"
            notas_p.append(f"{pos_final - (ref['pos'][-1] + ref['d_desliza']):+.2f} mm vs A")
        nota_p = "<br>".join(notas_p)
        return dict(
            slip=d['se_desliza_alguna_fase'], peak=d['g_max_fase'], mu=s["mu"], where=d['peor_fase_label'],
            perfil=perfil if comparar else None, color=color,
            figs=[("Safety factor", fmt_fs(d['factor_seguridad_fase']), "μ / peak"),
                  ("Cycle time", f"{s['t'][-1]:.2f} s", nota_t),
                  ("Final part position", f"{pos_final:.2f} mm", nota_p)],
        )

    filas = [fila_veredicto(A, "Profile A", COL_A)]
    if comparar:
        filas.append(fila_veredicto(B, "Profile B", COL_B, ref=A))
    panel_veredicto(filas)

    # ---- Configuration alerts ----
    if det_a['insufficient_distance']:
        aviso("warning",
            f"**Profile A — Inconsistent configuration:** the `RAMP_DECEL A` ramp needs "
            f"**{det_a['dist_needed_decel']:.1f} mm** to go from SPEED_AUTO_FAST down to SPEED_AUTO_SLOW, "
            f"but the `Slowdown Sensor Distance A` is only **{ss.sensor_distance_a:.1f} mm**."
        )
    if comparar and det_b['insufficient_distance']:
        aviso("warning",
            f"**Profile B — Inconsistent configuration:** the `RAMP_DECEL B` ramp needs "
            f"**{det_b['dist_needed_decel']:.1f} mm** to go from SPEED_AUTO_FAST down to SPEED_AUTO_SLOW, "
            f"but the `Slowdown Sensor Distance B` is only **{ss.sensor_distance_b:.1f} mm**."
        )
    if det_a['mechanically_capped']:
        aviso("info",
            f"**Profile A:** configured `RAMP_STOP A` ({ss.ramp_stop_a:.1f} mm/s²) exceeds the "
            f"achievable mechanical maximum ({det_a['a_mechanical_max']:.1f} mm/s², limited by the "
            f"{T_MIN_MECANICO_MS:.0f} ms floor). The mechanical limit was applied: actual deceleration = "
            f"**{det_a['a_stop_conveyor']:.1f} mm/s²** (≈{det_a['ramp_stop_real_ms']:.1f} ms)."
        )
    if comparar and det_b['mechanically_capped']:
        aviso("info",
            f"**Profile B:** configured `RAMP_STOP B` ({ss.ramp_stop_b:.1f} mm/s²) exceeds the "
            f"achievable mechanical maximum ({det_b['a_mechanical_max']:.1f} mm/s²). The mechanical limit was applied: "
            f"actual deceleration = **{det_b['a_stop_conveyor']:.1f} mm/s²** (≈{det_b['ramp_stop_real_ms']:.1f} ms)."
        )

    tab_mov, tab_slip, tab_ab, tab_fund = st.tabs([
        "Motion", "Slip", "Profile A vs B", "Fundamentals",
    ])

    # ------------------------------------------
    with tab_mov:
        st.plotly_chart(figura_movimiento_conveyor(A, B), theme=None)
        st.caption(
            "Labels: acceleration of each ramp in G, with the effective acceleration (mm/s²) and its duration — "
            "dark = within μ, red = exceeds μ, thick border = most critical phase. While cruising the acceleration "
            "is 0 G. Bands: Profile A PLC phases. Hover over the velocity curve to see the position x."
        )

        perfiles_zona = [("A", A, COL_A, "solid")] + ([("B", B, COL_B, "dashdot")] if comparar else [])
        fig_zona = figura_zona_paro(perfiles_zona)
        if fig_zona is not None:
            st.plotly_chart(fig_zona, theme=None)
        st.caption(
            "Zoom on the last millimeters. The arrows mark Δ, how far past the stop sensor each part rests, and "
            "the sensor move that cancels it (see Sensor correction below). Distance from the stop sensor to where the part rests ≈ "
            "SPEED_AUTO_SLOW² / (2 × the deceleration the part actually follows): the conveyor's stop "
            "deceleration, or μ·g if the part slides. A slower creep speed can therefore stop shorter "
            "even with a softer RAMP_STOP."
        )
        for nombre, s, _, _ in perfiles_zona:
            st.markdown(texto_zona_paro(s, f"Profile {nombre}" if comparar else "Profile A"))

        st.subheader("Sensor correction")
        st.caption(
            "Δ is how far past the stop sensor the part comes to rest. Moving the sensors back by Δ makes "
            "the part rest exactly where the stop sensor is today (Total Conveyor Length)."
        )
        for nombre, s, color, _ in perfiles_zona:
            k = nombre.lower()
            c = correccion_sensores(s, ss[f"sensor_distance_{k}"])
            if c is None:
                continue
            if comparar:
                clave_perfil(f"Profile {nombre}", color)
            m1, m2, m3 = st.columns(3)
            m1.metric("Stop offset Δ (part past the stop sensor)", f"{c['delta']:+.2f} mm".replace("-", "−"))
            m2.metric("Move the stop sensor to", f"{c['stop_nuevo']:.2f} mm",
                      delta=f"{-c['delta']:+.2f} mm", delta_color="off")
            m3.metric("Move the slowdown sensor to", f"{c['red_nuevo']:.2f} mm",
                      delta=f"{-c['delta']:+.2f} mm", delta_color="off",
                      help="Keeps the same Slowdown Sensor Distance, so the whole approach shifts by Δ.")
            notas = []
            if c["creep"] - c["delta"] >= 1.0:
                notas.append(f"Moving only the stop sensor also works: the creep stretch at SPEED_AUTO_SLOW "
                             f"({c['creep']:.1f} mm) is longer than Δ.")
            elif c["creep"] >= 0:
                notas.append(f"Move both sensors: the creep stretch at SPEED_AUTO_SLOW ({c['creep']:.1f} mm) leaves "
                             f"less than 1 mm of margin over Δ, so moving only the stop sensor could trigger the stop "
                             f"while the conveyor is still slowing down.")
            if s["desliza"]:
                notas.append(f"The part slips ({s['d_desliza']:.2f} mm), so where it rests depends on the real μ — "
                             f"the correction is only as accurate as μ. Lowering RAMP_STOP until it no longer "
                             f"slips makes the stop repeatable.")
            if notas:
                st.caption(" ".join(notas))

        st.subheader("Positioning & cycle time")

        def fila_posicion(s, nombre, color, ref=None):
            if comparar:
                clave_perfil(f"Profile {nombre}", color)

            def d(v, v_ref, fmt, unidad):
                return None if ref is None else f"{v - v_ref:+{fmt}} {unidad} vs A"

            pf = s["pos"][-1] + s["d_desliza"]
            pf_ref = None if ref is None else ref["pos"][-1] + ref["d_desliza"]
            c = st.columns(6)
            c[0].metric("Conveyor Overrun", f"{s['overrun']:.2f} mm",
                        delta=d(s["overrun"], ref and ref["overrun"], ".2f", "mm"), delta_color="inverse",
                        help="How far the conveyor travels past the stop sensor while it brakes.")
            c[1].metric("Overrun Time", f"{s['overrun_time']*1000:.0f} ms",
                        delta=d(s["overrun_time"] * 1000, ref and ref["overrun_time"] * 1000, ".0f", "ms"),
                        delta_color="inverse")
            c[2].metric("Conveyor Rests At", f"{s['pos'][-1]:.2f} mm",
                        delta=d(s["pos"][-1], ref and ref["pos"][-1], ".2f", "mm"), delta_color="off")
            c[3].metric("Part Slip", f"{s['d_desliza']:.2f} mm",
                        delta=d(s["d_desliza"], ref and ref["d_desliza"], ".2f", "mm"), delta_color="inverse",
                        help="Extra distance the part slides on the conveyor because the stop exceeds μ.")
            c[4].metric("Final Part Position", f"{pf:.2f} mm",
                        delta=d(pf, pf_ref, ".2f", "mm"), delta_color="off",
                        help="Where the conveyor rests plus the part slip.")
            c[5].metric("Total Motion Time", f"{s['t'][-1]:.2f} s",
                        delta=d(s["t"][-1], ref and ref["t"][-1], ".2f", "s"), delta_color="inverse")

        fila_posicion(A, "A", COL_A)
        if comparar:
            fila_posicion(B, "B", COL_B, ref=A)

        st.subheader("Cycle phases")
        filas = tabla_fases_conveyor(A, "A" if comparar else None)
        if comparar:
            filas += tabla_fases_conveyor(B, "B")
        st.dataframe(filas, hide_index=True)

    # ------------------------------------------
    with tab_slip:
        st.plotly_chart(figura_fases_conveyor(A, B), theme=None)

        perfiles_slip = [("A", "a", A, COL_A)] + ([("B", "b", B, COL_B)] if comparar else [])

        st.subheader("Acceleration in G per state transition")
        st.caption(
            "The linear conveyor has three speed transitions where the part experiences acceleration: "
            "start-up (ACCEL_FAST), braking to creep speed (DECEL_TO_SLOW), and the final stop "
            "(DECEL_TO_STOP). Each one is checked against μ separately."
        )
        for nombre, k, s, color in perfiles_slip:
            d = s["det"]
            mu = s["mu"]
            if comparar:
                clave_perfil(f"Profile {nombre}", color)
            fa1, fa2, fa3, fb1, fb2 = st.columns([1, 1, 1, 1, 1])
            fa1.metric("ACCEL_FAST (start-up)", f"{d['g_accel']:.3f} G",
                       delta=delta_mu(d['g_accel'], mu), delta_color="inverse")
            fa2.metric("DECEL_TO_SLOW (braking to creep)", f"{d['g_decel']:.3f} G",
                       delta=delta_mu(d['g_decel'], mu), delta_color="inverse")
            fa3.metric("DECEL_TO_STOP (final stop)", f"{s['g_conv']:.3f} G",
                       delta=delta_mu(s['g_conv'], mu), delta_color="inverse")
            fb1.metric("Most Critical Phase", d['peor_fase_label'].split(" (")[0], help=d['peor_fase_label'])
            fb2.metric("Safety Factor (most critical)", fmt_fs(d['factor_seguridad_fase']))

            prefijo = f"**Profile {nombre}** — " if comparar else ""
            if d['se_desliza_alguna_fase'] and not d['se_desliza_stop_fase']:
                aviso("warning",
                    f"{prefijo}**Difference detected:** the final stop is STABLE, but **{d['peor_fase_label']}** "
                    f"reaches {d['g_max_fase']:.3f} G, which **exceeds** μ ({mu:.2f} G)."
                )
            elif d['se_desliza_alguna_fase']:
                aviso("error", f"{prefijo}At least one phase exceeds μ — the most critical is **{d['peor_fase_label']}**.")
            else:
                aviso("success", f"{prefijo}All three speed transitions are within the friction limit.")

        st.divider()
        st.subheader("Inertia & part slip at the final stop")
        for nombre, k, s, color in perfiles_slip:
            d = s["det"]
            mu = s["mu"]
            if comparar:
                clave_perfil(f"Profile {nombre}", color)
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Conveyor Deceleration", f"{s['g_conv']:.3f} G")
            col2.metric("Friction Limit (μ)", f"{mu:.2f} G")
            col3.metric("Load Stability", estado_txt(s['desliza']),
                        delta=delta_mu(s['g_conv'], mu), delta_color="inverse")
            col4.metric("Relative Part Slip",
                        f"{s['d_desliza']:.3f} mm ({s['d_desliza']*1000:.0f} µm)" if s['desliza'] else "0.000 mm")
            col5.metric("Safety Factor (μ / g_conv)", fmt_fs(d['factor_seguridad']))
            col6.metric("Minimum μ to NOT slip", f"{d['mu_minimo_requerido']:.3f}")

        for nombre, k, s, color in perfiles_slip:
            d = s["det"]
            mu = s["mu"]
            v_slow = ss[f"speed_slow_{k}"]
            ramp_stop = ss[f"ramp_stop_{k}"]
            titulo = (f"View step-by-step calculation breakdown (Profile {nombre})" if s['desliza']
                      else f"View stability & deceleration math (Profile {nombre})")
            with st.expander(titulo):
                st.markdown("### Live Calculation Breakdown (Simulated Values)")

                st.markdown("**Step 1: Mechanical Maximum Deceleration (Floor Limit)**")
                st.latex(L(
                    f"a_{{~text{{mech,max}}}} = ~frac{{v_{{~text{{slow}}}}}}{{20~,~text{{ms}}}} = "
                    f"~frac{{{v_slow:.1f}~,~text{{mm/s}}}}{{0.020~,~text{{s}}}} = "
                    f"~mathbf{{{d['a_mechanical_max']:.1f}~,~text{{mm/s}}^2}}"
                ))

                st.markdown("**Step 2: Effective Stop Deceleration (RAMP_STOP capped by the mechanical limit)**")
                st.latex(L(
                    f"a_{{~text{{stop}}}} = ~min({ramp_stop:.1f}~,~text{{mm/s}}^2, "
                    f"{d['a_mechanical_max']:.1f}~,~text{{mm/s}}^2) = ~mathbf{{{d['a_stop_conveyor']:.1f}~,~text{{mm/s}}^2}} "
                    f"~quad (t~approx{d['ramp_stop_real_ms']:.1f}~,~text{{ms}})"
                ))
                st.latex(L(
                    f"g_{{~text{{conv}}}} = ~frac{{{d['a_stop_conveyor']:.1f}~,~text{{mm/s}}^2}}"
                    f"{{9810~,~text{{mm/s}}^2}} = ~mathbf{{{s['g_conv']:.3f}~,~text{{G}}}}"
                ))

                st.markdown("**Step 3: Maximum Allowable Friction Acceleration**")
                st.latex(L(
                    f"a_{{~text{{max~_piece}}}} = ~mu ~cdot g = {mu:.2f} ~cdot 9810~,~text{{mm/s}}^2 "
                    f"= {d['a_max_pieza']:.2f}~,~text{{mm/s}}^2 ~quad (~mu = ~mathbf{{{mu:.2f}~,~text{{G}}}})"
                ))

                st.markdown("**Step 4: Slip Decision Criterion**")
                if s['desliza']:
                    aviso("error", f"**SLIP DETECTED:** g_conv ({s['g_conv']:.3f} G) > μ ({mu:.2f} G).")
                    st.markdown("**Step 5: Relative Slip Distance (Δd)**")
                    st.latex(L(
                        f"d_{{~text{{piece}}}} = ~frac{{v_{{~text{{slow}}}}^2}}{{2 ~cdot a_{{~text{{max~_piece}}}}}} = "
                        f"{d['dist_freno_pieza']:.3f}~,~text{{mm}}"
                    ))
                    st.latex(L(
                        f"d_{{~text{{conveyor}}}} = ~frac{{v_{{~text{{slow}}}}^2}}{{2 ~cdot a_{{~text{{stop}}}}}} = "
                        f"{d['dist_freno_conveyor']:.3f}~,~text{{mm}}"
                    ))
                    st.latex(L(
                        f"~Delta d = {d['dist_freno_pieza']:.3f} - {d['dist_freno_conveyor']:.3f} = "
                        f"~mathbf{{{s['d_desliza']:.3f}~,~text{{mm}}}}"
                    ))
                else:
                    aviso("success", f"**STABLE LOAD:** g_conv ({s['g_conv']:.3f} G) ≤ μ ({mu:.2f} G).")

    # ------------------------------------------
    with tab_ab:
        if not comparar:
            aviso("info", "Turn on **Compare with Profile B** in the sidebar to compare two PLC configurations. "
                    "Profile B starts as a copy of Profile A.")
        else:
            st.caption("Every key value of both profiles side by side. B − A is B minus A.")
            ajustes, acel, paro = tablas_comparacion(A, B, ss)
            st.subheader("Stop & positioning")
            st.dataframe(paro, hide_index=True)
            st.subheader("Acceleration on the part")
            st.dataframe(acel, hide_index=True)
            st.subheader("Settings")
            st.dataframe(ajustes, hide_index=True)

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
    st.header("Physics Engine & Kinematic Formulas — Conveyor")
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
    st.header("Physics Engine & Kinematic Formulas — Turntable")
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
