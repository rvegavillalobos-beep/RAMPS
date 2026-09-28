import streamlit as st
import numpy as np
import plotly.graph_objects as go

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Kinematic & Physics Conveyor Simulator",
    layout="wide"
)
st.title("Kinematic & Physics Conveyor Simulator")
st.markdown("Real time analysis of velocity profile, positioning, and inertial part slip.")

# SESSION STATE INITIALIZATION
defaults = {
    "conveyor_length": 3000.0,
    "speed_fast_a": 300.0,
    "speed_slow_a": 100.0,
    "accel_a": 300.0,
    "decel_a": 300.0,
    "sensor_distance_a": 150.0,
    "ramp_stop_a": 150.0,
    "mu_a": 0.28,
    "comparar": False,
    "speed_fast_b": 450.0,
    "speed_slow_b": 120.0,
    "accel_b": 400.0,
    "decel_b": 300.0,
    "sensor_distance_b": 150.0,
    "ramp_stop_b": 0.0,
    "mu_b": 0.28,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "_comparar_prev" not in st.session_state:
    st.session_state._comparar_prev = False

# SIDEBAR
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

# --- Copia A -> B automáticamente al activar la comparación ---
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
    dt = 0.001  # 1 ms
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
    t_sensor_red = 0.0
    t_sensor_stop = 0.0

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
                state = "CRUISE_FAST"
        elif state == "CRUISE_FAST":
            if p >= pos_sensor_red:
                t_sensor_red = t[i]
                state = "DECEL_TO_SLOW"
        elif state == "DECEL_TO_SLOW":
            v -= decel * dt
            if v <= v_slow:
                v = v_slow
                state = "CRUISE_SLOW"
        elif state == "CRUISE_SLOW":
            if p >= pos_stop:
                t_sensor_stop = t[i]
                state = "DECEL_TO_STOP"
        elif state == "DECEL_TO_STOP":
            v -= a_stop_conveyor * dt
            if v <= 0:
                v = 0.0
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
    overrun_time = t[-1] - t_sensor_stop if t_sensor_stop > 0 else 0.0

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

    details = {
        "ramp_stop_real_ms": ramp_stop_real_ms,
        "a_stop_conveyor": a_stop_conveyor,
        "a_max_pieza": a_max_pieza,
        "dist_freno_pieza": dist_freno_pieza,
        "dist_freno_conveyor": dist_freno_conveyor,
    }

    return t, pos, vel, t_sensor_red, t_sensor_stop, dist_overrun_conveyor, overrun_time, g_conveyor, g_pieza_real, se_desliza, deslizamiento_mm, details


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
        xaxis_title="Time (s)",
        yaxis_title="Velocity (mm/s)",
        height=460,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(b=110)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 2. SLIP ANALYSIS (justo debajo de la gráfica principal) ----
    st.markdown("---")
    st.subheader("📊 Inertia & Part Slip Analysis (Profile A)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conveyor Deceleration", f"{g_conv_a:.3f} G")
    col2.metric("Friction Limit (μ)", f"{st.session_state.mu_a:.2f} G")
    col3.metric("Load Stability", "🔴 SLIPPING" if desliza_a else "🟢 STABLE")
    col4.metric("Relative Part Slip", f"{d_desliza_a:.2f} mm" if desliza_a else "0.00 mm")

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
    col6.metric("Final Conveyor Position", f"{pos_a[-1]:.2f} mm")
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
    # 5. ANIMACIÓN TIPO "BARRA DE PROGRESO" (AL FINAL DE LA PÁGINA)
    # ==========================================
    st.markdown("---")
    st.subheader("🎬 Animación en Tiempo Real: Recorrido de la Pieza")
    st.caption("La barra se rellena de color conforme la pieza avanza. Usa Play/Pause o el slider para explorar el recorrido a detalle.")

    def construir_animacion_barra(perfiles, length, n_frames=180):
        t_end = max(p['t'][-1] for p in perfiles)
        frame_times = np.linspace(0, t_end, n_frames)

        for p in perfiles:
            p['pos_i'] = np.interp(frame_times, p['t'], p['pos'])
            p['vel_i'] = np.interp(frame_times, p['t'], p['vel'])

        n_perfiles = len(perfiles)
        x_max = length * 1.22

        fig_bar = go.Figure()

        annotations_static = [
            dict(x=0, y=1.14, xref="x", yref="paper", text="<b>Start</b>",
                 showarrow=False, font=dict(size=12, color="gray")),
            dict(x=length, y=1.14, xref="x", yref="paper", text="<b>Ziel</b>",
                 showarrow=False, font=dict(size=12, color="#444")),
        ]

        for idx, p in enumerate(perfiles):
            row = idx
            color = p['color']

            param_text = (f"<b>{p['label']}</b><br>"
                          f"v_fast={p['v_fast']:.0f} mm/s | v_slow={p['v_slow']:.0f} mm/s<br>"
                          f"t_ciclo={p['t'][-1]:.2f} s")
            annotations_static.append(
                dict(x=0.0, y=row, xref="paper", yref="y", text=param_text,
                     showarrow=False, align="left", xanchor="left",
                     font=dict(size=11, color="#333"), xshift=-15)
            )

            # Overrun annotation (solo para el primer perfil / Perfil A)
            if 'overrun_mm' in p:
                annotations_static.append(
                    dict(x=length, y=row + 0.38, xref="x", yref="y",
                         text=f"<b>Overrun: +{p['overrun_mm']:.1f} mm  (+{p['overrun_time']*1000:.0f} ms)</b>",
                         showarrow=True, arrowhead=2, ax=40, ay=-25,
                         font=dict(size=11, color="#b00000"),
                         bgcolor="rgba(255,235,235,0.9)", bordercolor="#b00000")
                )

            # Track de fondo (gris, estático)
            fig_bar.add_trace(go.Bar(
                x=[length], y=[row], base=[0], orientation='h', width=0.5,
                marker_color="rgba(220,220,220,0.6)", showlegend=False, hoverinfo='skip'
            ))
            # Barra rellena (se anima)
            fig_bar.add_trace(go.Bar(
                x=[p['pos_i'][0]], y=[row], base=[0], orientation='h', width=0.5,
                marker_color=color, showlegend=False, hoverinfo='skip', opacity=0.85
            ))
            # Marcador cuadrado en la punta (se anima)
            fig_bar.add_trace(go.Scatter(
                x=[p['pos_i'][0]], y=[row], mode='markers',
                marker=dict(size=20, color=color, symbol='square', line=dict(color='black', width=2)),
                showlegend=False, hoverinfo='skip'
            ))
            # Lectura dinámica a la derecha (se anima)
            pct0 = 100.0 * p['pos_i'][0] / length
            fig_bar.add_trace(go.Scatter(
                x=[x_max * 0.98], y=[row], mode='text',
                text=[f"<b>{p['pos_i'][0]:.0f} mm</b><br>{p['vel_i'][0]:.1f} mm/s<br>{pct0:.0f}%"],
                textposition="middle left", showlegend=False, hoverinfo='skip',
                textfont=dict(size=11, color="#333")
            ))

        frames = []
        for k in range(n_frames):
            frame_data = []
            trace_indices = []
            for idx, p in enumerate(perfiles):
                base_idx = idx * 4
                filled_idx = base_idx + 1
                marker_idx = base_idx + 2
                text_idx = base_idx + 3

                pos_k = p['pos_i'][k]
                vel_k = p['vel_i'][k]
                pct_k = 100.0 * pos_k / length

                frame_data.append(go.Bar(x=[pos_k], y=[idx], base=[0]))
                trace_indices.append(filled_idx)

                frame_data.append(go.Scatter(x=[pos_k], y=[idx]))
                trace_indices.append(marker_idx)

                frame_data.append(go.Scatter(
                    x=[x_max * 0.98], y=[idx],
                    text=[f"<b>{pos_k:.0f} mm</b><br>{vel_k:.1f} mm/s<br>{pct_k:.0f}%"]
                ))
                trace_indices.append(text_idx)

            frames.append(go.Frame(data=frame_data, traces=trace_indices, name=str(k)))

        fig_bar.frames = frames
        frame_ms = max(1000 * (frame_times[1] - frame_times[0]), 20) if n_frames > 1 else 50

        fig_bar.update_layout(
            height=160 + n_perfiles * 150,
            template="plotly_white",
            barmode='overlay',
            xaxis=dict(title="Posición (mm)", range=[-length * 0.05, x_max], zeroline=False),
            yaxis=dict(visible=False, range=[-0.6, n_perfiles - 0.4 + 0.6]),
            annotations=annotations_static,
            margin=dict(l=220, t=70, b=70, r=110),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.20, x=0.0, xanchor="left",
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
                x=0.0, len=0.96, y=-0.08
            )]
        )
        return fig_bar

    perfiles_animacion = [
        {
            "label": "Perfil A",
            "color": "#1f77b4",
            "t": t_a, "pos": pos_a, "vel": vel_a,
            "v_fast": st.session_state.speed_fast_a,
            "v_slow": st.session_state.speed_slow_a,
            "overrun_mm": overrun_a,
            "overrun_time": overrun_time_a,
        }
    ]
    if st.session_state.comparar:
        perfiles_animacion.append({
            "label": "Perfil B",
            "color": "#d62728",
            "t": t_b, "pos": pos_b, "vel": vel_b,
            "v_fast": st.session_state.speed_fast_b,
            "v_slow": st.session_state.speed_slow_b,
        })

    fig_bar = construir_animacion_barra(perfiles_animacion, st.session_state.conveyor_length)
    st.plotly_chart(fig_bar, use_container_width=True)


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

    st.subheader("5. Integration Engine & Kinematic Profiles")
    st.markdown(
        "The state machine integrates velocity and position at a step size of $\\\\Delta t = 1\\\\,\\\\text{ms}$:\\n\\n"
        "1. **ACCEL_FAST**: Accelerates at RAMP_ACCEL up to SPEED_AUTO_FAST.\\n"
        "2. **CRUISE_FAST**: Maintains fast cruise speed until reaching $P_{\\\\text{reduction}} = L_{\\\\text{total}} - S_{\\\\text{distance}}$.\\n"
        "3. **DECEL_TO_SLOW**: Decelerates at RAMP_DECEL down to SPEED_AUTO_SLOW.\\n"
        "4. **CRUISE_SLOW**: Creeps at slow speed until reaching $P_{\\\\text{stop}} = L_{\\\\text{total}}$.\\n"
        "5. **DECEL_TO_STOP**: Final stop deceleration based on $t_{\\\\text{stop\\\\_real}}$."
    )
