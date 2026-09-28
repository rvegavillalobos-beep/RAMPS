import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Kinematic & Physics Conveyor Simulator", layout="wide")
st.title("Kinematic & Physics Conveyor Simulator")
st.markdown("Real time analysis of velocity profile, positioning, and inertial part slip.")

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

    for i in range(1, steps):
        t[i] = t[i - 1] + dt

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
            if p >= pos_stop:
                t_sensor_stop = t[i]
                state = "DECEL_TO_STOP"
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

    # Safety KPIs
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
    }

    return (t, pos, vel, t_sensor_red, t_sensor_stop, dist_overrun_conveyor, overrun_time,
            g_conveyor, g_pieza_real, se_desliza, deslizamiento_mm, details)


# SIMULATION EXECUTION
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

# TABS
tab_sim, tab_math = st.tabs(["📊 Simulation & Dashboard", "📚 Mathematical Background & Physics Engine"])

# ==========================================
# TAB 1: SIMULATION & DASHBOARD
# ==========================================
with tab_sim:

    # ---- 1. GRÁFICA DE VELOCIDAD ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Velocity Profile A',
                              line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=[t_red_a, t_red_a], y=[0, max(vel_a) * 1.1], mode='lines',
                              name='Slowdown Sensor A', line=dict(color="#ff7f0e", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=[t_stop_a, t_stop_a], y=[0, max(vel_a) * 1.1], mode='lines',
                              name='Stop Sensor A', line=dict(color="#d62728", width=2, dash="dash")))
    if st.session_state.comparar:
        fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Velocity Profile B',
                                  line=dict(color='#9467bd', width=3, dash='dashdot')))

    fig.update_layout(
        title="Velocity Profile (mm/s) vs Time (s)",
        xaxis_title="Time (s)", yaxis_title="Velocity (mm/s)",
        height=460, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(b=110)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 2. SLIP ANALYSIS ----
    st.markdown("---")
    st.subheader("📊 Inertia & Part Slip Analysis (Profile A)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conveyor Deceleration", f"{g_conv_a:.3f} G")
    col2.metric("Friction Limit (μ)", f"{st.session_state.mu_a:.2f} G")
    col3.metric("Load Stability", "🔴 SLIPPING" if desliza_a else "🟢 STABLE")
    col4.metric("Relative Part Slip", f"{d_desliza_a:.2f} mm" if desliza_a else "0.00 mm")

    col_s1, col_s2 = st.columns(2)
    fs = det_a['factor_seguridad']
    fs_display = "∞" if fs == float('inf') else f"{fs:.2f}x"
    col_s1.metric("🛡️ Factor de Seguridad (μ / g_conv)", fs_display,
                  help="Si es menor a 1.0x, la pieza desliza. Mayor a 1.0x = margen de seguridad.")
    col_s2.metric("🎯 μ mínimo requerido para NO deslizar", f"{det_a['mu_minimo_requerido']:.3f}",
                  help="Coeficiente de fricción mínimo entre pieza y banda necesario para evitar el deslizamiento con esta configuración.")

    expander_title = "🔍 View Calculation Step by Step Breakdown (Profile A)" if desliza_a else "ℹ️ View Stability & Deceleration Math (Profile A)"
    with st.expander(expander_title):
        st.markdown("### 🧮 Live Calculation Breakdown (Simulated Values)")
        st.markdown("**Step 1: Effective Stop Ramp Time**")
        st.latex(rf"t_{{\\text{{stop\\_real}}}} = \\max({st.session_state.ramp_stop_a:.1f}\\,\\text{{ms}}, 20.0\\,\\text{{ms}}) = {det_a['ramp_stop_real_ms']:.1f}\\,\\text{{ms}} = {det_a['ramp_stop_real_ms']/1000.0:.3f}\\,\\text{{s}}")
        st.markdown("**Step 2: Conveyor Stop Deceleration**")
        st.latex(rf"a_{{\\text{{stop}}}} = \\frac{{{st.session_state.speed_slow_a:.1f}\\,\\text{{mm/s}}}}{{{det_a['ramp_stop_real_ms']/1000.0:.3f}\\,\\text{{s}}}} = {det_a['a_stop_conveyor']:.2f}\\,\\text{{mm/s}}^2")
        st.latex(rf"g_{{\\text{{conv}}}} = \\frac{{{det_a['a_stop_conveyor']:.2f}\\,\\text{{mm/s}}^2}}{{9810\\,\\text{{mm/s}}^2}} = \\mathbf{{{g_conv_a:.3f}\\,\\text{{G}}}}")
        st.markdown("**Step 3: Maximum Allowable Friction Acceleration**")
        st.latex(rf"a_{{\\text{{max\\_piece}}}} = \\mu \\cdot g = {st.session_state.mu_a:.2f} \\cdot 9810\\,\\text{{mm/s}}^2 = {det_a['a_max_pieza']:.2f}\\,\\text{{mm/s}}^2 \\quad (\\mu = \\mathbf{{{st.session_state.mu_a:.2f}\\,\\text{{G}}}})")
        st.markdown("**Step 4: Slip Decision Criteria**")
        if desliza_a:
            st.error(f"🔴 **SLIP DETECTED:** $g_{{\\\\text{{conv}}}}$ ({g_conv_a:.3f} G) > μ ({st.session_state.mu_a:.2f} G). Stopping force exceeds static friction capacity!")
            st.markdown("**Step 5: Relative Slip Distance Calculation (Δd)**")
            st.latex(rf"d_{{\\text{{piece}}}} = \\frac{{v_{{\\text{{slow}}}}^2}}{{2 \\cdot a_{{\\text{{max\\_piece}}}}}} = \\frac{{{st.session_state.speed_slow_a:.1f}^2}}{{2 \\cdot {det_a['a_max_pieza']:.2f}}} = {det_a['dist_freno_pieza']:.2f}\\,\\text{{mm}}")
            st.latex(rf"d_{{\\text{{conveyor}}}} = \\frac{{v_{{\\text{{slow}}}}^2}}{{2 \\cdot a_{{\\text{{stop}}}}}} = \\frac{{{st.session_state.speed_slow_a:.1f}^2}}{{2 \\cdot {det_a['a_stop_conveyor']:.2f}}} = {det_a['dist_freno_conveyor']:.2f}\\,\\text{{mm}}")
            st.latex(rf"\\Delta d = d_{{\\text{{piece}}}} - d_{{\\text{{conveyor}}}} = {det_a['dist_freno_pieza']:.2f} - {det_a['dist_freno_conveyor']:.2f} = \\mathbf{{{d_desliza_a:.2f}\\,\\text{{mm}}}}")
        else:
            st.success(f"🟢 **STABLE LOAD:** $g_{{\\\\text{{conv}}}}$ ({g_conv_a:.3f} G) ≤ μ ({st.session_state.mu_a:.2f} G). Friction holds the part securely. Relative Part Slip = **0.00 mm**.")

    # ---- 3. Positioning & Cycle Time ----
    st.markdown("---")
    st.subheader("🎯 Positioning & Cycle Time (Profile A)")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Conveyor Overrun", f"{overrun_a:.2f} mm")
    col6.metric("Overrun Time", f"{overrun_time_a*1000:.0f} ms")
    col7.metric("Final Part Position", f"{(pos_a[-1] + d_desliza_a):.2f} mm")
    col8.metric("Total Motion Time", f"{t_a[-1]:.2f} s")

    # ---- 4. Comparación A vs B ----
    if st.session_state.comparar:
        st.markdown("---")
        st.subheader("⚖️ Profile A vs Profile B Comparison")
        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        c_b1.metric("Deceleration Profile B", f"{g_conv_b:.3f} G", delta=f"{(g_conv_b - g_conv_a):.3f} G", delta_color="inverse")
        c_b2.metric("Part Slip Profile B", f"{d_desliza_b:.2f} mm", delta=f"{(d_desliza_b - d_desliza_a):.2f} mm", delta_color="inverse")
        c_b3.metric("Final Part Pos B", f"{(pos_b[-1] + d_desliza_b):.2f} mm")
        c_b4.metric("Cycle Time Delta", f"{t_b[-1]:.2f} s", delta=f"{(t_b[-1] - t_a[-1]):.2f} s", delta_color="inverse")

    # ==========================================
    # 5. ANIMACIÓN — TRACK VISUAL (AL FINAL DE LA PÁGINA)
    # ==========================================
    st.markdown("---")
    st.subheader("🎬 Animación en Tiempo Real: Recorrido de la Pieza")
    st.caption("El riel se colorea conforme la pieza avanza. El color y la etiqueta cambian según la fase cinemática activa (aceleración, crucero, frenado).")

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        calidad = st.selectbox("Calidad de animación", ["Baja (ligera)", "Media", "Alta (detallada)"], index=1)
    with ctrl2:
        velocidad_reproduccion = st.select_slider("Velocidad de reproducción", options=["0.5x", "1x", "2x", "4x"], value="1x")

    n_frames_map = {"Baja (ligera)": 90, "Media": 160, "Alta (detallada)": 260}
    speed_map = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
    n_frames = n_frames_map[calidad]
    speed_mult = speed_map[velocidad_reproduccion]

    STATE_INFO = {
        "ACCEL_FAST":    ("🚀", "Acelerando"),
        "CRUISE_FAST":   ("➡️", "Crucero Rápido"),
        "DECEL_TO_SLOW": ("🐢", "Frenando a Lento"),
        "CRUISE_SLOW":   ("🐌", "Crucero Lento"),
        "DECEL_TO_STOP": ("🛑", "Frenado Final"),
        "DONE":          ("✅", "Detenido"),
    }
    STATE_COLOR = {
        "ACCEL_FAST": "#2ecc71", "CRUISE_FAST": None,  # None -> usa color de carril
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

        # ---- Leyenda explicativa (trazos invisibles solo para leyenda) ----
        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="#ff7f0e", dash="dot", width=2), name="Sensor Reducción"))
        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="#d62728", dash="dash", width=2), name="Ziel (Sensor Paro)"))
        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="rgba(220,50,50,0.35)", size=15, symbol='square'), name="Zona Overrun / Slip"))
        fig_t.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol="triangle-down", size=12, color="black"), name="Alto Real (posición final)"))

        # ---- Riel de fondo y elementos estáticos por carril ----
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

        # ---- Trazos animados (fill + marcador + texto en vivo) por carril ----
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
        base_ms = (1000 * (frame_times[1] - frame_times[0])) if n_frames > 1 else 50
        frame_ms = max(base_ms / speed_mult, 15)

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
                         args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                           fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(k)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True))],
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
# TAB 2: MATHEMATICAL & PHYSICAL BACKGROUND
# ==========================================
with tab_math:
    st.header("📐 Physics Engine & Kinematic Formulas")
    st.markdown(
        "This section details the analytical models used to simulate conveyor dynamics, "
        "deceleration forces, friction boundaries, and inertial displacement of transported parts."
    )

    st.subheader("1. Mechanical Elasticity Floor")
    st.markdown(
        "Industrial belt/roller conveyors exhibit mechanical compliance (chain slack, belt stretch, and chassis flex). "
        "Even if PLC parameters define a stopping ramp near $0\\\\text{ ms}$, the physical mechanical response time "
        "is lower bounded by $T_{\\\\text{min}} = 20.0\\\\,\\\\text{ms}$:"
    )
    st.latex(r"t_{\\text{stop\\_real}} = \\max\\left(t_{\\text{ramp\\_stop}}, 20.0\\,\\text{ms}\\right)")

    st.subheader("2. Conveyor Stop Deceleration")
    st.markdown("When the part trips the stop sensor at creep velocity $v_{\\\\text{slow}}$, the conveyor applies a stopping deceleration $a_{\\\\text{stop}}$:")
    st.latex(r"a_{\\text{stop}} = \\frac{v_{\\text{slow}}}{t_{\\text{stop\\_real}}}")
    st.markdown("Expressed in dimensionless $G$ forces relative to $g = 9810\\\\,\\\\text{mm/s}^2$:")
    st.latex(r"g_{\\text{conv}} = \\frac{a_{\\text{stop}}}{9810}")

    st.subheader("3. Static Friction Threshold & Slip Determination")
    st.markdown("According to Coulomb's Law of Dry Friction, the maximum shear force transmitted without slipping is:")
    st.latex(r"F_{\\text{friction\\_max}} = \\mu \\cdot m \\cdot g")
    st.latex(r"a_{\\text{max\\_piece}} = \\mu \\cdot g = \\mu \\cdot 9810\\,\\text{mm/s}^2")
    st.latex(r"g_{\\text{max\\_piece}} = \\mu")
    st.markdown(
        "* If $g_{\\\\text{conv}} \\\\le \\\\mu$: The static friction force holds the part in place. **No slip occurs** ($\\\\Delta d = 0$).\\n"
        "* If $g_{\\\\text{conv}} > \\\\mu$: The stopping force exceeds static friction limits. The part **slips forward by inertia**."
    )

    st.subheader("4. Relative Part Slip Estimation")
    st.latex(r"d_{\\text{piece}} = \\frac{v_{\\text{slow}}^2}{2 \\cdot a_{\\text{max\\_piece}}}")
    st.latex(r"d_{\\text{conveyor}} = \\frac{v_{\\text{slow}}^2}{2 \\cdot a_{\\text{stop}}}")
    st.latex(r"\\Delta d = d_{\\text{piece}} - d_{\\text{conveyor}} = \\frac{v_{\\text{slow}}^2}{2} \\left( \\frac{1}{\\mu \\cdot g} - \\frac{1}{a_{\\text{stop}}} \\right)")

    st.subheader("5. Safety Factor & Minimum Required Friction")
    st.markdown("A practical engineering KPI to gauge margin before slip occurs:")
    st.latex(r"\\text{Factor de Seguridad} = \\frac{\\mu}{g_{\\text{conv}}}")
    st.markdown(
        "* $> 1.0$: hay margen — la fricción disponible supera la fuerza de frenado.\\n"
        "* $= 1.0$: límite crítico exacto.\\n"
        "* $< 1.0$: la pieza **desliza** con la configuración actual."
    )
    st.latex(r"\\mu_{\\text{minimo requerido}} = g_{\\text{conv}}")

    st.subheader("6. Integration Engine & Kinematic Profiles")
    st.markdown(
        "The state machine integrates velocity and position at a step size of $\\\\Delta t = 1\\\\,\\\\text{ms}$:\\n\\n"
        "1. **ACCEL_FAST**: Accelerates at RAMP_ACCEL up to SPEED_AUTO_FAST.\\n"
        "2. **CRUISE_FAST**: Maintains fast cruise speed until reaching $P_{\\\\text{reduction}} = L_{\\\\text{total}} - S_{\\\\text{distance}}$.\\n"
        "3. **DECEL_TO_SLOW**: Decelerates at RAMP_DECEL down to SPEED_AUTO_SLOW.\\n"
        "4. **CRUISE_SLOW**: Creeps at slow speed until reaching $P_{\\\\text{stop}} = L_{\\\\text{total}}$.\\n"
        "5. **DECEL_TO_STOP**: Final stop deceleration based on $t_{\\\\text{stop\\\\_real}}$."
    )
