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
CARRIER_ALTO_MM = 250.0    # 25 cm - Work Piece Carrier (va sobre las ruedas del conveyor, sin slip)
CARRIER_LARGO_MM = 1600.0  # 160 cm
CARGA_ALTO_MM = 130.0      # 13 cm - Pieza (sujeta solo por fricción sobre el carrier)
CARGA_LARGO_MM = 1200.0    # 120 cm

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


# ==========================================
# REALISTIC KINEMATIC CALCULATION ENGINE
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

    # ---- VALIDACIÓN DE CONFIGURACIÓN: distancia de sensor vs rampa de decel ----
    # Distancia física mínima necesaria para bajar de v_fast a v_slow con la
    # rampa RAMP_DECEL configurada. Si la Distancia Sensor Reducción es menor
    # a esto, el sistema llegará al sensor de paro SIN haber alcanzado v_slow.
    if v_fast > v_slow and decel > 0:
        dist_needed_decel = (v_fast ** 2 - v_slow ** 2) / (2.0 * decel)
    else:
        dist_needed_decel = 0.0
    insufficient_distance = dist_needed_decel > s_dist

    for i in range(1, steps):
        t[i] = t[i - 1] + dt

        # ---- CHEQUEO UNIVERSAL DEL SENSOR DE PARO (FIX) ----
        # El sensor de paro es un evento físico de POSICIÓN: se activa apenas
        # la posición alcanza pos_stop, sin importar en qué estado cinemático
        # se encuentre el sistema (incluso si todavía está en DECEL_TO_SLOW
        # porque la distancia del sensor de reducción fue insuficiente).
        # Antes este chequeo SOLO se hacía dentro de CRUISE_SLOW, lo cual
        # causaba que RAMP_DECEL contaminara incorrectamente el overrun.
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
            pass  # el único evento relevante aquí (llegar a pos_stop) ya se cubrió arriba
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

    # Si nunca se llegó a v_slow (por distancia insuficiente), t_reach_slow
    # queda en 0.0; lo igualamos a t_sensor_stop para que la clasificación de
    # fases en las animaciones no confunda ese tramo con CRUISE_SLOW.
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

tab_sim, tab_math = st.tabs(["📊 Simulation & Dashboard", "📚 Mathematical Background & Physics Engine"])

# ==========================================
# TAB 1: SIMULATION & DASHBOARD
# ==========================================
with tab_sim:

    # ---- AVISO DE CONFIGURACIÓN: distancia de sensor insuficiente ----
    if det_a['insufficient_distance']:
        st.warning(
            f"⚠️ **Perfil A — Configuración inconsistente:** la rampa `RAMP_DECEL A` necesita "
            f"**{det_a['dist_needed_decel']:.1f} mm** para bajar de SPEED_AUTO_FAST a SPEED_AUTO_SLOW, "
            f"pero la `Distancia Sensor Reducción A` solo tiene **{st.session_state.sensor_distance_a:.1f} mm**. "
            f"La pieza llegará al sensor de paro sin haber alcanzado la velocidad de creep, causando un overrun "
            f"mayor al esperado. Aumenta la distancia del sensor o reduce RAMP_DECEL / SPEED_AUTO_FAST."
        )
    if st.session_state.comparar and det_b['insufficient_distance']:
        st.warning(
            f"⚠️ **Perfil B — Configuración inconsistente:** la rampa `RAMP_DECEL B` necesita "
            f"**{det_b['dist_needed_decel']:.1f} mm** para bajar de SPEED_AUTO_FAST a SPEED_AUTO_SLOW, "
            f"pero la `Distancia Sensor Reducción B` solo tiene **{st.session_state.sensor_distance_b:.1f} mm**. "
            f"La pieza llegará al sensor de paro sin haber alcanzado la velocidad de creep, causando un overrun "
            f"mayor al esperado. Aumenta la distancia del sensor o reduce RAMP_DECEL / SPEED_AUTO_FAST."
        )

    # ---- 1. GRÁFICA DE VELOCIDAD ----
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
    col_s1.metric("🛡️ Factor de Seguridad (μ / g_conv)", fs_display,
                  help="Si es menor a 1.0x, la pieza desliza. Mayor a 1.0x = margen de seguridad.")
    col_s2.metric("🎯 μ mínimo requerido para NO deslizar", f"{det_a['mu_minimo_requerido']:.3f}",
                  help="Coeficiente de fricción mínimo entre la pieza y el carrier necesario para evitar el deslizamiento con esta configuración.")

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
                f"🔴 **SLIP DETECTED:** g_conv ({g_conv_a:.3f} G) > μ ({st.session_state.mu_a:.2f} G). "
                f"La fuerza de frenado del carrier supera la capacidad de fricción estática de la pieza!"
            )
            st.markdown("**Step 5: Relative Slip Distance Calculation (Δd)**")
            st.latex(L(
                f"d_{{~text{{piece}}}} = ~frac{{v_{{~text{{slow}}}}^2}}{{2 ~cdot a_{{~text{{max~_piece}}}}}} = "
                f"~frac{{{st.session_state.speed_slow_a:.1f}^2}}{{2 ~cdot {det_a['a_max_pieza']:.2f}}} = "
                f"{det_a['dist_freno_pieza']:.3f}~,~text{{mm}}"
            ))
            st.latex(L(
                f"d_{{~text{{conveyor}}}} = ~frac{{v_{{~text{{slow}}}}^2}}{{2 ~cdot a_{{~text{{stop}}}}}} = "
                f"~frac{{{st.session_state.speed_slow_a:.1f}^2}}{{2 ~cdot {det_a['a_stop_conveyor']:.2f}}} = "
                f"{det_a['dist_freno_conveyor']:.3f}~,~text{{mm}}"
            ))
            st.latex(L(
                f"~Delta d = d_{{~text{{piece}}}} - d_{{~text{{conveyor}}}} = "
                f"{det_a['dist_freno_pieza']:.3f} - {det_a['dist_freno_conveyor']:.3f} = "
                f"~mathbf{{{d_desliza_a:.3f}~,~text{{mm}}}}"
            ))
        else:
            st.success(
                f"🟢 **STABLE LOAD:** g_conv ({g_conv_a:.3f} G) ≤ μ ({st.session_state.mu_a:.2f} G). "
                f"La fricción sostiene la pieza sobre el carrier sin deslizar. Relative Part Slip = **0.000 mm**."
            )

    # ---- 3. Positioning & Cycle Time ----
    st.markdown("---")
    st.subheader("🎯 Positioning & Cycle Time (Profile A)")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Conveyor Overrun", f"{overrun_a:.2f} mm")
    col6.metric("Overrun Time", f"{overrun_time_a*1000:.0f} ms")
    col7.metric("Final Part Position", f"{(pos_a[-1] + d_desliza_a):.2f} mm")
    col8.metric("Total Motion Time", f"{t_a[-1]:.2f} s")

    # ---- 4. Comparación A vs B (con Delta de Overrun) ----
    if st.session_state.comparar:
        st.markdown("---")
        st.subheader("⚖️ Profile A vs Profile B Comparison")

        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        c_b1.metric("Deceleration Profile B", f"{g_conv_b:.3f} G",
                    delta=f"{(g_conv_b - g_conv_a):.3f} G", delta_color="inverse")
        c_b2.metric("Part Slip Profile B", f"{d_desliza_b:.3f} mm",
                    delta=f"{(d_desliza_b - d_desliza_a):.3f} mm", delta_color="inverse")
        c_b3.metric("Overrun Profile B", f"{overrun_b:.2f} mm",
                    delta=f"{(overrun_b - overrun_a):.2f} mm", delta_color="inverse")
        c_b4.metric("Overrun Time Profile B", f"{overrun_time_b*1000:.0f} ms",
                    delta=f"{(overrun_time_b - overrun_time_a)*1000:.0f} ms", delta_color="inverse")

        c_b5, c_b6 = st.columns(2)
        c_b5.metric("Final Part Pos B", f"{(pos_b[-1] + d_desliza_b):.2f} mm",
                    delta=f"{(pos_b[-1] + d_desliza_b) - (pos_a[-1] + d_desliza_a):.2f} mm", delta_color="inverse")
        c_b6.metric("Cycle Time Profile B", f"{t_b[-1]:.2f} s",
                    delta=f"{(t_b[-1] - t_a[-1]):.2f} s", delta_color="inverse")

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
            data_k = []
            idx_k = []
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
                if is_slip:
                    txt += " ⚠️ SLIP"
                txt_color = "#e74c3c" if is_slip else "#333"

                fi = base_traces + idx * 3
                data_k.append(go.Scatter(x=[0, pos_k], y=[row, row], line=dict(color=marker_color)))
                idx_k.append(fi)
                data_k.append(go.Scatter(x=[pos_k], y=[row], marker=dict(color=marker_color)))
                idx_k.append(fi + 1)
                data_k.append(go.Scatter(x=[length * 1.02], y=[row], text=[txt], textfont=dict(color=txt_color)))
                idx_k.append(fi + 2)

            frames.append(go.Frame(data=data_k, traces=idx_k, name=str(k)))

        fig_t.frames = frames
        target_total_ms = float(np.clip(t_end * 1000.0, 4000.0, 16000.0))
        frame_ms = max((target_total_ms / n_frames) / speed_mult, 8.0)

        fig_t.update_layout(
            height=200 + n_lanes * 190,
            template="plotly_white",
            xaxis=dict(title="Posición (mm)", range=[-length * 0.02, x_max]),
            yaxis=dict(visible=False, range=[-0.7, max(rows) + 0.7]),
            margin=dict(l=230, t=70, b=60, r=40),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.22, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=frame_ms, redraw=False),
                                           fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(k)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=False))],
                            label=f"{frame_times[k]:.1f}s") for k in range(n_frames)],
                x=0.0, len=0.96, y=-0.32
            )]
        )
        return fig_t

    def g_por_fase(accel, decel, g_conv):
        return {
            "ACCEL_FAST": accel / 9810.0,
            "CRUISE_FAST": 0.0,
            "DECEL_TO_SLOW": decel / 9810.0,
            "CRUISE_SLOW": 0.0,
            "DECEL_TO_STOP": g_conv,
            "DONE": 0.0,
        }

    perfiles_animacion = [{
        "label": "Perfil A", "color": "#1f77b4",
        "t": t_a, "pos": pos_a, "vel": vel_a,
        "v_fast": st.session_state.speed_fast_a, "v_slow": st.session_state.speed_slow_a,
        "accel": st.session_state.accel_a, "decel": st.session_state.decel_a, "mu": st.session_state.mu_a,
        "pos_sensor_red": det_a['pos_sensor_red'],
        "t_reach_fast": det_a['t_reach_fast'], "t_red": t_red_a, "t_reach_slow": det_a['t_reach_slow'],
        "t_stop": t_stop_a, "t_full": det_a['t_fully_stopped'],
        "overrun_mm": overrun_a, "overrun_ms": overrun_time_a * 1000, "slip_mm": d_desliza_a,
        "se_desliza": desliza_a, "g_por_fase": g_por_fase(st.session_state.accel_a, st.session_state.decel_a, g_conv_a),
    }]
    if st.session_state.comparar:
        perfiles_animacion.append({
            "label": "Perfil B", "color": "#d62728",
            "t": t_b, "pos": pos_b, "vel": vel_b,
            "v_fast": st.session_state.speed_fast_b, "v_slow": st.session_state.speed_slow_b,
            "accel": st.session_state.accel_b, "decel": st.session_state.decel_b, "mu": st.session_state.mu_b,
            "pos_sensor_red": det_b['pos_sensor_red'],
            "t_reach_fast": det_b['t_reach_fast'], "t_red": t_red_b, "t_reach_slow": det_b['t_reach_slow'],
            "t_stop": t_stop_b, "t_full": det_b['t_fully_stopped'],
            "overrun_mm": overrun_b, "overrun_ms": overrun_time_b * 1000, "slip_mm": d_desliza_b,
            "se_desliza": desliza_b, "g_por_fase": g_por_fase(st.session_state.accel_b, st.session_state.decel_b, g_conv_b),
        })

    fig_track = construir_track(perfiles_animacion, st.session_state.conveyor_length, n_frames, speed_mult)
    st.plotly_chart(fig_track, use_container_width=True)

    # ==========================================
    # 6. WORK PIECE CARRIER + CARGA (EJE FIJO — SIN CÁMARA MÓVIL)
    # ==========================================
    st.markdown("---")
    st.subheader("🔬 Work Piece Carrier & Carga: Deslizamiento en Vivo")
    st.caption(
        f"El Carrier (160 × 25 cm, gris) va rígidamente sobre las ruedas del conveyor — nunca desliza. "
        f"La Carga (120 × 13 cm, color) va encima sujeta solo por fricción. Ambos viajan juntos. "
        f"El eje, el grid y las líneas de sensores son fijos: el único elemento que se mueve es el conjunto Carrier + Carga."
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        velocidad_carrier = st.select_slider("Velocidad de reproducción", options=["0.5x", "1x", "2x", "4x"], value="1x", key="vel_carrier")
    with col_ctrl2:
        exageracion_txt = st.select_slider(
            "Exageración visual del deslizamiento",
            options=["1x (real)", "5x", "10x", "25x", "50x"], value="10x", key="exag_slip",
            help="El deslizamiento real entre carrier y carga suele ser de fracciones de milímetro, "
                 "invisible a escala real. Este control amplifica SOLO el desplazamiento relativo para "
                 "poder verlo — el valor numérico mostrado siempre es el real, nunca el exagerado."
        )
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
            carrier_pos_i = np.interp(frame_times, p['t'], p['pos'])  # rígido: sigue al conveyor siempre
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
                load_pos_i = carrier_pos_i.copy()  # se mueve exactamente igual, slip = 0 siempre

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
            fig_c.add_shape(type="line", x0=x_min_fixed, x1=x_max_fixed, y0=y0c, y1=y0c,
                             line=dict(color="#bbbbbb", width=2))
            fig_c.add_shape(type="line", x0=p['pos_sensor_red'], x1=p['pos_sensor_red'],
                             y0=y0c - 40, y1=y0c + carrier_alto + carga_alto + 60,
                             line=dict(color="#ff7f0e", width=2, dash="dot"))
            fig_c.add_shape(type="line", x0=length, x1=length,
                             y0=y0c - 40, y1=y0c + carrier_alto + carga_alto + 60,
                             line=dict(color="#d62728", width=2, dash="dash"))

            info_txt = (f"<b>{p['label']}</b><br>"
                        f"v_slow={p['v_slow']:.0f} mm/s · μ={p['mu']:.2f}<br>"
                        f"Slip máx real: {p['slip_i'].max():.3f} mm")
            fig_c.add_annotation(x=0.0, y=y0c + (carrier_alto + carga_alto) / 2, xref="paper", yref="y",
                                  xanchor="left", text=info_txt, showarrow=False, align="left",
                                  font=dict(size=11, color="#333"), xshift=-18)

        base_traces = len(fig_c.data)
        for p in perfiles:
            y0c = p['y_off']
            cf0 = p['carrier_pos_i'][0]
            lf0 = p['lf_i'][0]
            fig_c.add_trace(go.Scatter(
                x=[cf0 - carrier_largo, cf0, cf0, cf0 - carrier_largo, cf0 - carrier_largo],
                y=[y0c, y0c, y0c + carrier_alto, y0c + carrier_alto, y0c],
                mode='lines', fill='toself', fillcolor="#95a5a6",
                line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'
            ))
            fig_c.add_trace(go.Scatter(
                x=[lf0 - carga_largo, lf0, lf0, lf0 - carga_largo, lf0 - carga_largo],
                y=[y0c + carrier_alto, y0c + carrier_alto, y0c + carrier_alto + carga_alto,
                   y0c + carrier_alto + carga_alto, y0c + carrier_alto],
                mode='lines', fill='toself', fillcolor=p['color'],
                line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'
            ))
            fig_c.add_trace(go.Scatter(x=[cf0], y=[y0c + carrier_alto + carga_alto + 35], mode='text',
                                        text=["..."], textfont=dict(size=11, color="#333"),
                                        showlegend=False, hoverinfo='skip'))

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
                data_k.append(go.Scatter(
                    x=[cf - carrier_largo, cf, cf, cf - carrier_largo, cf - carrier_largo],
                    y=[y0c, y0c, y0c + carrier_alto, y0c + carrier_alto, y0c]
                ))
                idx_k.append(fi)
                data_k.append(go.Scatter(
                    x=[lf - carga_largo, lf, lf, lf - carga_largo, lf - carga_largo],
                    y=[y0c + carrier_alto, y0c + carrier_alto, y0c + carrier_alto + carga_alto,
                       y0c + carrier_alto + carga_alto, y0c + carrier_alto],
                    fillcolor=load_color
                ))
                idx_k.append(fi + 1)

                exag_note = f" (mostrado a {exageracion:.0f}x)" if exageracion > 1 and slip_now > 0 else ""
                txt = f"Carrier: {cf:.0f} mm | Slip real: {slip_now:.3f} mm ({slip_now*1000:.0f} µm){exag_note}"
                if is_slip_now:
                    txt = "⚠️ DESLIZANDO — " + txt
                data_k.append(go.Scatter(x=[cf], y=[y0c + carrier_alto + carga_alto + 35],
                                          text=[txt], textfont=dict(color="#e74c3c" if is_slip_now else "#333")))
                idx_k.append(fi + 2)

            frames.append(go.Frame(data=data_k, traces=idx_k, name=str(k)))

        fig_c.frames = frames
        target_total_ms = float(np.clip((t_end - t_start) * 1000.0, 4000.0, 14000.0))
        frame_ms = max((target_total_ms / n_frames) / speed_mult, 8.0)

        fig_c.update_layout(
            height=180 + n_lanes * (carrier_alto / 2 + carga_alto + 240),
            template="plotly_white",
            xaxis=dict(title="Posición (mm) — eje fijo", range=[x_min_fixed, x_max_fixed]),
            yaxis=dict(visible=False, range=[-60, n_lanes * lane_height + 60]),
            margin=dict(l=200, t=70, b=60, r=40),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.15, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=frame_ms, redraw=False),
                                           fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(k)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=False))],
                            label=f"{frame_times[k]:.2f}s") for k in range(n_frames)],
                x=0.0, len=0.96, y=-0.30
            )]
        )
        return fig_c

    perfiles_carrier = [{
        "label": "Perfil A", "color": "#1f77b4",
        "t": t_a, "pos": pos_a,
        "v_slow": st.session_state.speed_slow_a, "mu": st.session_state.mu_a,
        "a_max_pieza": det_a['a_max_pieza'],
        "se_desliza": desliza_a, "dist_freno_pieza": det_a['dist_freno_pieza'],
        "t_stop": t_stop_a, "t_red": t_red_a, "pos_sensor_red": det_a['pos_sensor_red'],
        "t_full": det_a['t_fully_stopped'],
    }]
    if st.session_state.comparar:
        perfiles_carrier.append({
            "label": "Perfil B", "color": "#d62728",
            "t": t_b, "pos": pos_b,
            "v_slow": st.session_state.speed_slow_b, "mu": st.session_state.mu_b,
            "a_max_pieza": det_b['a_max_pieza'],
            "se_desliza": desliza_b, "dist_freno_pieza": det_b['dist_freno_pieza'],
            "t_stop": t_stop_b, "t_red": t_red_b, "pos_sensor_red": det_b['pos_sensor_red'],
            "t_full": det_b['t_fully_stopped'],
        })

    fig_carrier = construir_carrier_load(perfiles_carrier, st.session_state.conveyor_length,
                                          n_frames_carrier, speed_mult_carrier, exageracion)
    st.plotly_chart(fig_carrier, use_container_width=True)


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
    st.markdown(L(
        "Industrial belt/roller conveyors exhibit mechanical compliance (chain slack, belt stretch, and chassis flex). "
        "Even if PLC parameters define a stopping ramp near $0~text{ ms}$, the physical mechanical response time "
        "is lower bounded by $T_{~text{min}} = 20.0~,~text{ms}$:"
    ))
    st.latex(L(r"t_{~text{stop~_real}} = ~max~left(t_{~text{ramp~_stop}}, 20.0~,~text{ms}~right)"))

    st.subheader("2. Conveyor Stop Deceleration")
    st.markdown(L("When the part trips the stop sensor at creep velocity $v_{~text{slow}}$, the conveyor applies a stopping deceleration $a_{~text{stop}}$:"))
    st.latex(L(r"a_{~text{stop}} = ~frac{v_{~text{slow}}}{t_{~text{stop~_real}}}"))
    st.markdown(L("Expressed in dimensionless $G$ forces relative to $g = 9810~,~text{mm/s}^2$:"))
    st.latex(L(r"g_{~text{conv}} = ~frac{a_{~text{stop}}}{9810}"))

    st.subheader("3. Static Friction Threshold & Slip Determination")
    st.markdown(L("According to Coulomb's Law of Dry Friction, the maximum shear force transmitted without slipping is governed by the static friction coefficient $~mu$:"))
    st.latex(L(r"F_{~text{friction~_max}} = ~mu ~cdot m ~cdot g"))
    st.latex(L(r"a_{~text{max~_piece}} = ~mu ~cdot g = ~mu ~cdot 9810~,~text{mm/s}^2"))
    st.latex(L(r"g_{~text{max~_piece}} = ~mu"))
    st.markdown(L(
        "**Slip Condition Criterion:**\\n"
        "* If $g_{~text{conv}} ~le ~mu$: the static friction force holds the part in place. **No slip occurs** ($~Delta d = 0$).\\n"
        "* If $g_{~text{conv}} > ~mu$: the stopping force exceeds static friction limits. The part breaks traction and **slips forward by inertia**."
    ))

    st.subheader("4. Relative Part Slip Estimation")
    st.markdown(L("When slip occurs, the conveyor decelerates at $a_{~text{stop}}$, while the part decelerates at a slower rate dictated solely by dynamic friction $a_{~text{max~_piece}}$."))
    st.markdown("Stopping distance of the part under friction:")
    st.latex(L(r"d_{~text{piece}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{max~_piece}}}"))
    st.markdown("Stopping distance of the physical conveyor belt:")
    st.latex(L(r"d_{~text{conveyor}} = ~frac{v_{~text{slow}}^2}{2 ~cdot a_{~text{stop}}}"))
    st.markdown(L("Net relative slippage displacement ($~Delta d$):"))
    st.latex(L(r"~Delta d = d_{~text{piece}} - d_{~text{conveyor}} = ~frac{v_{~text{slow}}^2}{2} ~left( ~frac{1}{~mu ~cdot g} - ~frac{1}{a_{~text{stop}}} ~right)"))

    st.subheader("5. Safety Factor & Minimum Required Friction")
    st.markdown("A practical engineering KPI to gauge margin before slip occurs:")
    st.latex(L(r"~text{Factor de Seguridad} = ~frac{~mu}{g_{~text{conv}}}"))
    st.markdown(
        "* $> 1.0$: hay margen — la fricción disponible supera la fuerza de frenado.\\n"
        "* $= 1.0$: límite crítico exacto.\\n"
        "* $< 1.0$: la pieza **desliza** con la configuración actual."
    )
    st.latex(L(r"~mu_{~text{minimo requerido}} = g_{~text{conv}}"))

    st.subheader("6. Carrier vs. Load: Two-Body Slip Model")
    st.markdown(L(
        "The simulation models **two rigid bodies traveling together**:\\n\\n"
        "* **Work Piece Carrier**: mechanically coupled to the conveyor rollers/wheels — it always follows the belt's "
        "exact kinematic profile, including the final stop deceleration $a_{~text{stop}}$. It never slips relative to the conveyor.\\n"
        "* **Load**: rests on top of the carrier, held in place **only by dry friction**. It moves in lockstep with the "
        "carrier during ACCEL_FAST, CRUISE_FAST, DECEL_TO_SLOW and CRUISE_SLOW. Only during DECEL_TO_STOP, if "
        "$g_{~text{conv}} > ~mu$, the load cannot keep up and slides forward relative to the carrier."
    ))
    st.latex(L(r"x_{~text{carrier}}(t) = x_{~text{conveyor}}(t) ~quad ~text{(rigid coupling, no slip, always)}"))
    st.latex(L(
        r"x_{~text{load}}(t) = ~begin{cases} x_{~text{carrier}}(t) & t < t_{~text{stop}} ~text{ or no slip} \\\\ "
        r"L_{~text{total}} + v_{~text{slow}}(t-t_{~text{stop}}) - ~frac{1}{2}a_{~text{max~_piece}}(t-t_{~text{stop}})^2 & "
        r"t ~ge t_{~text{stop}} ~text{ and slip active} ~end{cases}"
    ))
    st.latex(L(r"~text{Slip}(t) = x_{~text{load}}(t) - x_{~text{carrier}}(t) ~ge 0"))
    st.markdown(
        "En condiciones típicas (μ≈0.28, velocidades de creep bajas), este deslizamiento resulta del orden de "
        "**fracciones de milímetro a pocos milímetros** — imperceptible a escala real, por lo que la animación "
        "incluye un factor de exageración visual aplicado únicamente al desplazamiento relativo (nunca a la posición absoluta)."
    )

    st.subheader("7. Sensor Distance Validation (Bug Fix)")
    st.markdown(L(
        "The distance between the reduction sensor and the stop sensor ($S_{~text{distance}}$) must be large enough "
        "for the belt to decelerate from $v_{~text{fast}}$ to $v_{~text{slow}}$ using the configured RAMP_DECEL:"
    ))
    st.latex(L(r"d_{~text{needed}} = ~frac{v_{~text{fast}}^2 - v_{~text{slow}}^2}{2 ~cdot ~text{RAMP~_DECEL}}"))
    st.markdown(L(
        "If $d_{~text{needed}} > S_{~text{distance}}$, the system reaches the stop sensor **before** finishing the "
        "deceleration ramp — the position trigger for the stop sensor is evaluated globally at every time step "
        "(regardless of kinematic state) so the final stop deceleration always engages exactly at $L_{~text{total}}$, "
        "preventing RAMP_DECEL from incorrectly contaminating the overrun calculation."
    ))

    st.subheader("8. Integration Engine & Kinematic Profiles")
    st.markdown(L(
        "The state machine integrates velocity and position at a step size of $~Delta t = 1~,~text{ms}$:\\n\\n"
        "1. **ACCEL_FAST**: Accelerates at RAMP_ACCEL up to SPEED_AUTO_FAST.\\n"
        "2. **CRUISE_FAST**: Maintains fast cruise speed until reaching $P_{~text{reduction}} = L_{~text{total}} - S_{~text{distance}}$.\\n"
        "3. **DECEL_TO_SLOW**: Decelerates at RAMP_DECEL down to SPEED_AUTO_SLOW.\\n"
        "4. **CRUISE_SLOW**: Creeps at slow speed until reaching $P_{~text{stop}} = L_{~text{total}}$.\\n"
        "5. **DECEL_TO_STOP**: Final stop deceleration based on $t_{~text{stop~_real}}$, triggered at $L_{~text{total}}$ "
        "regardless of current kinematic state."
    ))
