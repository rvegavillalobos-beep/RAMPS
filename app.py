import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kinematic & Physics Conveyor Simulator",
    layout="wide"
)

st.title("Kinematic & Physics Conveyor Simulator")
st.markdown("Real-time analysis of velocity profile, positioning, and inertial part slip.")

# --- SESSION STATE INITIALIZATION ---
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
    "mu_b": 0.28
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- SIDEBAR (PARAMETERS - LABELS PRESERVED) ---
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

if st.session_state.comparar:
    st.sidebar.header("🟣 Perfil B (Comparativa)")
    st.sidebar.number_input("SPEED_AUTO_FAST B (mm/s)", value=st.session_state.speed_fast_b, step=10.0, key="speed_fast_b")
    st.sidebar.number_input("SPEED_AUTO_SLOW B (mm/s)", value=st.session_state.speed_slow_b, step=10.0, key="speed_slow_b")
    st.sidebar.number_input("RAMP_ACCEL B (mm/s²)", value=st.session_state.accel_b, step=50.0, key="accel_b")
    st.sidebar.number_input("RAMP_DECEL B (mm/s²)", value=st.session_state.decel_b, step=50.0, key="decel_b")
    st.sidebar.number_input("Distancia Sensor Reducción B (mm)", value=st.session_state.sensor_distance_b, step=25.0, key="sensor_distance_b")
    st.sidebar.number_input("RAMP_STOP B (ms)", value=st.session_state.ramp_stop_b, step=10.0, min_value=0.0, key="ramp_stop_b")
    st.sidebar.number_input("Coeficiente Fricción μ B", value=st.session_state.mu_b, step=0.01, min_value=0.01, max_value=1.0, key="mu_b")


# --- REALISTIC KINEMATIC CALCULATION ENGINE ---
def calcular_perfil(v_fast, v_slow, accel, decel, length, s_dist, ramp_stop_ms, mu):
    dt = 0.001  # Integration time step of 1 ms
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
    
    # 1. Mechanical elasticity limit (minimum 20 ms due to chain backlash, chassis flex, and gear slack)
    T_MIN_MECANICO_MS = 20.0
    ramp_stop_real_ms = max(ramp_stop_ms, T_MIN_MECANICO_MS)
    
    # 2. Real conveyor deceleration at stop sensor (mm/s² and Gs)
    a_stop_conveyor = (v_slow / (ramp_stop_real_ms / 1000.0))
    g_conveyor = (a_stop_conveyor / 1000.0) / 9.81
    
    # 3. Acceleration limit due to static friction (mm/s² and Gs)
    g_max_pieza = mu
    a_max_pieza = mu * 9.81 * 1000.0
    
    for i in range(1, steps):
        t[i] = t[i-1] + dt
        
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
        
        # Truncate array upon motion completion
        if state == "DONE" and i > 50 and np.all(vel[i-20:i] == 0):
            t = t[:i+1]
            pos = pos[:i+1]
            vel = vel[:i+1]
            break
            
    # Distance traveled by conveyor after hitting stop sensor
    dist_overrun_conveyor = pos[-1] - pos_stop
    
    # 4. Kinematic calculation of part slip (mm)
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
        "dist_freno_conveyor": dist_freno_conveyor
    }
        
    return t, pos, vel, t_sensor_red, t_sensor_stop, dist_overrun_conveyor, g_conveyor, g_pieza_real, se_desliza, deslizamiento_mm, details


# --- SIMULATION EXECUTION ---
t_a, pos_a, vel_a, t_red_a, t_stop_a, overrun_a, g_conv_a, g_pieza_a, desliza_a, d_desliza_a, det_a = calcular_perfil(
    st.session_state.speed_fast_a, st.session_state.speed_slow_a, 
    st.session_state.accel_a, st.session_state.decel_a, 
    st.session_state.conveyor_length, st.session_state.sensor_distance_a,
    st.session_state.ramp_stop_a, st.session_state.mu_a
)

if st.session_state.comparar:
    t_b, pos_b, vel_b, t_red_b, t_stop_b, overrun_b, g_conv_b, g_pieza_b, desliza_b, d_desliza_b, det_b = calcular_perfil(
        st.session_state.speed_fast_b, st.session_state.speed_slow_b, 
        st.session_state.accel_b, st.session_state.decel_b, 
        st.session_state.conveyor_length, st.session_state.sensor_distance_b,
        st.session_state.ramp_stop_b, st.session_state.mu_b
    )

# --- TABS FOR UI STRUCTURE ---
tab_sim, tab_math = st.tabs(["📊 Simulation & Dashboard", "📚 Mathematical Background & Physics Engine"])

# ==========================================
# TAB 1: SIMULATION & DASHBOARD
# ==========================================
with tab_sim:
    # Interactive Plots
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Velocity Profile (mm/s)", "Position Trajectory (mm)"),
        horizontal_spacing=0.10
    )

    # Plot 1: Velocity
    fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Velocity Profile A', line=dict(color='#1f77b4', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t_red_a, t_red_a], y=[0, max(vel_a)*1.1], mode='lines', name='Slowdown Sensor A', line=dict(color="#ff7f0e", width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t_stop_a, t_stop_a], y=[0, max(vel_a)*1.1], mode='lines', name='Stop Sensor A', line=dict(color="#d62728", width=2, dash="dash")), row=1, col=1)

    if st.session_state.comparar:
        fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Velocity Profile B', line=dict(color='#9467bd', width=3, dash='dashdot')), row=1, col=1)

    # Plot 2: Position
    fig.add_trace(go.Scatter(x=t_a, y=pos_a, mode='lines', name='Conveyor Position A', line=dict(color='#2ca02c', width=3)), row=1, col=2)
    fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=st.session_state.conveyor_length, y1=st.session_state.conveyor_length, line=dict(color="#d62728", width=2, dash="dash"), row=1, col=2)
    fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=st.session_state.conveyor_length-st.session_state.sensor_distance_a, y1=st.session_state.conveyor_length-st.session_state.sensor_distance_a, line=dict(color="#ff7f0e", width=2, dash="dot"), row=1, col=2)

    if st.session_state.comparar:
        fig.add_trace(go.Scatter(x=t_b, y=pos_b, mode='lines', name='Conveyor Position B', line=dict(color='#8c564b', width=3, dash='dashdot')), row=1, col=2)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (mm/s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Position (mm)", row=1, col=2)

    fig.update_layout(
        height=480, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(b=120)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics Panel: Inertia & Friction
    st.markdown("---")
    st.subheader("📊 Inertia & Part Slip Analysis (Profile A)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conveyor Deceleration", f"{g_conv_a:.3f} G")
    col2.metric("Friction Limit (μ)", f"{st.session_state.mu_a:.2f} G")
    col3.metric("Load Stability", "🔴 SLIPPING" if desliza_a else "🟢 STABLE")
    col4.metric("Relative Part Slip", f"{d_desliza_a:.2f} mm" if desliza_a else "0.00 mm")

    # Dynamic Expander for Live Calculations Breakdown
    expander_title = "🔍 View Calculation Step-by-Step Breakdown (Profile A)" if desliza_a else "ℹ️ View Stability & Deceleration Math (Profile A)"
    with st.expander(expander_title):
        st.markdown("### 🧮 Live Calculation Breakdown (Simulated Values)")
        
        st.markdown(f"**Step 1: Effective Stop Ramp Time ($t_{{\\text{{stop\_real}}}}$)**")
        st.latex(rf"t_{{\text{{stop\_real}}}} = \max({st.session_state.ramp_stop_a:.1f}\,\text{{ms}}, 20.0\,\text{{ms}}) = {det_a['ramp_stop_real_ms']:.1f}\,\text{{ms}} = {det_a['ramp_stop_real_ms']/1000.0:.3f}\,\text{{s}}")
        
        st.markdown(f"**Step 2: Conveyor Stop Deceleration ($a_{{\\text{{stop}} rational}}$ & $g_{{\\text{{conv}}}}$)**")
        st.latex(rf"a_{{\text{{stop}}}} = \frac{{\text{{SPEED\_AUTO\_SLOW A}}}}{{t_{{\text{{stop\_real}}}}}} = \frac{{{st.session_state.speed_slow_a:.1f}\,\text{{mm/s}}}}{{{det_a['ramp_stop_real_ms']/1000.0:.3f}\,\text{{s}}}} = {det_a['a_stop_conveyor']:.2f}\,\text{{mm/s}}^2")
        st.latex(rf"g_{{\text{{conv}}}} = \frac{{{det_a['a_stop_conveyor']:.2f}\,\text{{mm/s}}^2}}{{9810\,\text{{mm/s}}^2}} = \mathbf{{{g_conv_a:.3f}\,\text{{G}}}}")

        st.markdown(f"**Step 3: Maximum Allowable Friction Acceleration ($a_{{\\text{{max\_piece}}}}$ & $\mu$)**")
        st.latex(rf"a_{{\text{{max\_piece}}}} = \mu \cdot g = {st.session_state.mu_a:.2f} \cdot 9810\,\text{{mm/s}}^2 = {det_a['a_max_pieza']:.2f}\,\text{{mm/s}}^2 \quad (\mu = \mathbf{{{st.session_state.mu_a:.2f}\,\text{{G}}}})")

        st.markdown(f"**Step 4: Slip Decision Criteria**")
        if desliza_a:
            st.error(f"🔴 **SLIP DETECTED:** $g_{{\\text{{conv}}}} ({g_conv_a:.3f}\,\text{{G}}) > \mu ({st.session_state.mu_a:.2f}\,\text{{G}})$. Stopping force exceeds static friction capacity!")
            
            st.markdown(f"**Step 5: Relative Slip Distance Calculation ($\Delta d$)**")
            st.latex(rf"d_{{\text{{piece}}}} = \frac{{v_{{\text{{slow}}}}^2}}{{2 \cdot a_{{\text{{max\_piece}}}}}} = \frac{{{st.session_state.speed_slow_a:.1f}^2}}{{2 \cdot {det_a['a_max_pieza']:.2f}}} = {det_a['dist_freno_pieza']:.2f}\,\text{{mm}}")
            st.latex(rf"d_{{\text{{conveyor}}}} = \frac{{v_{{\text{{slow}}}}^2}}{{2 \cdot a_{{\text{{stop}}}}}} = \frac{{{st.session_state.speed_slow_a:.1f}^2}}{{2 \cdot {det_a['a_stop_conveyor']:.2f}}} = {det_a['dist_freno_conveyor']:.2f}\,\text{{mm}}")
            st.latex(rf"\Delta d = d_{{\text{{piece}}}} - d_{{\text{{conveyor}}}} = {det_a['dist_freno_pieza']:.2f}\,\text{{mm}} - {det_a['dist_freno_conveyor']:.2f}\,\text{{mm}} = \mathbf{{{d_desliza_a:.2f}\,\text{{mm}}}}")
        else:
            st.success(f"🟢 **STABLE LOAD:** $g_{{\\text{{conv}}}} ({g_conv_a:.3f}\,\text{{G}}) \le \mu ({st.session_state.mu_a:.2f}\,\text{{G}})$. Friction holds the part securely. Relative Part Slip = **0.00 mm**.")

    # Metrics Panel: Positioning & Timing
    st.markdown("---")
    st.subheader("🎯 Positioning & Cycle Time (Profile A)")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Conveyor Overrun", f"{overrun_a:.2f} mm")
    col6.metric("Final Conveyor Position", f"{pos_a[-1]:.2f} mm")
    col7.metric("Final Part Position", f"{(pos_a[-1] + d_desliza_a):.2f} mm")
    col8.metric("Total Motion Time", f"{t_a[-1]:.2f} s")

    # Comparison Metrics (Profile B)
    if st.session_state.comparar:
        st.markdown("---")
        st.subheader("⚖️ Profile A vs Profile B Comparison")
        
        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        c_b1.metric("Deceleration Profile B", f"{g_conv_b:.3f} G", delta=f"{(g_conv_b - g_conv_a):.3f} G", delta_color="inverse")
        c_b2.metric("Part Slip Profile B", f"{d_desliza_b:.2f} mm", delta=f"{(d_desliza_b - d_desliza_a):.2f} mm", delta_color="inverse")
        c_b3.metric("Final Part Pos B", f"{(pos_b[-1] + d_desliza_b):.2f} mm")
        c_b4.metric("Cycle Time Delta", f"{t_b[-1]:.2f} s", delta=f"{(t_b[-1] - t_a[-1]):.2f} s", delta_color="inverse")


# ==========================================
# TAB 2: MATHEMATICAL & PHYSICAL BACKGROUND
# ==========================================
with tab_math:
    st.header("📐 Physics Engine & Kinematic Formulas")
    st.markdown(
        "This section details the analytical models used to simulate conveyor dynamics, "
        "deceleration forces, friction boundaries, and inertial displacement of transported parts."
    )
    
    st.markdown("---")
    st.subheader("1. Mechanical Elasticity Floor")
    st.markdown(
        "Industrial belt/roller conveyors exhibit mechanical compliance (chain slack, belt stretch, and chassis flex). "
        "Even if PLC parameters define a stopping ramp near $0\\text{ ms}$, the physical mechanical response time is lower-bounded by $T_{\\text{min}} = 20.0\\text{ ms}$:"
    )
    st.latex(r"t_{\text{stop\_real}} = \max\left(t_{\text{ramp\_stop}}, 20.0\,\text{ms}\right)")

    st.markdown("---")
    st.subheader("2. Conveyor Stop Deceleration")
    st.markdown(
        "When the part trips the stop sensor at creep velocity $v_{\\text{slow}}$, the conveyor applies a stopping deceleration $a_{\\text{stop}}$:"
    )
    st.latex(r"a_{\text{stop}} = \frac{v_{\text{slow}}}{t_{\text{stop\_real}}}")
    st.markdown("Expressed in dimensionless $G$-forces relative to gravitational acceleration $g = 9.81\,\text{m/s}^2$ ($9810\,\text{mm/s}^2$):")
    st.latex(r"g_{\text{conv}} = \frac{a_{\text{stop}}}{9810}")

    st.markdown("---")
    st.subheader("3. Static Friction Threshold & Slip Determination")
    st.markdown(
        "According to Coulomb's Law of Dry Friction, the maximum horizontal shear force transmitted without slipping is governed by the static friction coefficient $\mu$:"
    )
    st.latex(r"F_{\text{friction\_max}} = \mu \cdot m \cdot g")
    st.latex(r"a_{\text{max\_piece}} = \mu \cdot g = \mu \cdot 9810\,\text{mm/s}^2")
    st.latex(r"g_{\text{max\_piece}} = \mu")

    st.markdown("**Slip Condition Criterion:**")
    st.markdown(
        "* If $g_{\text{conv}} \le \mu$: The static friction force holds the part in place. **No slip occurs** ($\Delta d = 0$).\n"
        "* If $g_{\text{conv}} > \mu$: The stopping force exceeds static friction limits. The part breaks traction and **slips forward by inertia**."
    )

    st.markdown("---")
    st.subheader("4. Relative Part Slip Estimation")
    st.markdown(
        "When slip occurs, the conveyor decelerates at $a_{\\text{stop}}$, while the part decelerates at a slower rate dictated solely by dynamic friction $a_{\\text{max\_piece}}$."
    )
    st.markdown("Stopping distance of the part under friction:")
    st.latex(r"d_{\text{piece}} = \frac{v_{\text{slow}}^2}{2 \cdot a_{\text{max\_piece}}}")
    st.markdown("Stopping distance of the physical conveyor belt:")
    st.latex(r"d_{\text{conveyor}} = \frac{v_{\text{slow}}^2}{2 \cdot a_{\text{stop}}}")
    st.markdown("Net relative slippage displacement ($\Delta d$):")
    st.latex(r"\Delta d = d_{\text{piece}} - d_{\text{conveyor}} = \frac{v_{\text{slow}}^2}{2} \left( \frac{1}{\mu \cdot g} - \frac{1}{a_{\text{stop}}} \right)")

    st.markdown("---")
    st.subheader("5. Integration Engine & Kinematic Profiles")
    st.markdown(
        "The state-machine integrates velocity and position at a step size of $\Delta t = 1\,\text{ms}$ through the following states:"
    )
    st.markdown(
        "1. **ACCEL_FAST**: Accelerates at $\\text{RAMP\_ACCEL}$ up to $\\text{SPEED\_AUTO\_FAST}$.\n"
        "2. **CRUISE_FAST**: Maintains fast cruise speed until reaching $P_{\\text{reduction}} = L_{\\text{total}} - S_{\\text{distance}}$.\n"
        "3. **DECEL_TO_SLOW**: Decelerates at $\\text{RAMP\_DECEL}$ down to $\\text{SPEED\_AUTO\_SLOW}$.\n"
        "4. **CRUISE_SLOW**: Creeps at slow speed until reaching $P_{\\text{stop}} = L_{\\text{total}}$.\n"
        "5. **DECEL_TO_STOP**: Emergency/final stop deceleration based on $t_{\\text{stop\_real}}$."
    )
