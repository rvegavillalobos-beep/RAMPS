import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Conveyor Kinematic & Payload Stability Simulator",
    layout="wide"
)

st.title("Conveyor Kinematic & Payload Stability Simulator")
st.markdown("Real-time kinematic profile simulation, overtravel positioning analysis, and dynamic payload slip prediction.")

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
    
    "enable_comparison": False,
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

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("📐 System Geometry")
st.sidebar.number_input("Total Conveyor Length (mm)", value=st.session_state.conveyor_length, step=100.0, key="conveyor_length")

st.sidebar.header("🔵 Motion Profile A (Primary)")
st.sidebar.number_input("Traverse Speed - FAST (mm/s)", value=st.session_state.speed_fast_a, step=10.0, key="speed_fast_a")
st.sidebar.number_input("Approach Speed - CREEP (mm/s)", value=st.session_state.speed_slow_a, step=10.0, key="speed_slow_a")
st.sidebar.number_input("Acceleration Rate (mm/s²)", value=st.session_state.accel_a, step=50.0, key="accel_a")
st.sidebar.number_input("Deceleration Rate (mm/s²)", value=st.session_state.decel_a, step=50.0, key="decel_a")
st.sidebar.number_input("Slowdown Photoeye Offset (mm)", value=st.session_state.sensor_distance_a, step=25.0, key="sensor_distance_a")
st.sidebar.number_input("Drive Stop Ramp Time (ms)", value=st.session_state.ramp_stop_a, step=10.0, min_value=0.0, key="ramp_stop_a", help="VFD / Servo deceleration ramp time upon reaching the stop photoeye.")
st.sidebar.number_input("Surface Friction Coefficient (μ)", value=st.session_state.mu_a, step=0.01, min_value=0.01, max_value=1.0, key="mu_a")

st.sidebar.markdown("---")
st.sidebar.checkbox("Enable Profile B Comparison", value=st.session_state.enable_comparison, key="enable_comparison")

if st.session_state.enable_comparison:
    st.sidebar.header("🟣 Motion Profile B (Comparison)")
    st.sidebar.number_input("Traverse Speed - FAST B (mm/s)", value=st.session_state.speed_fast_b, step=10.0, key="speed_fast_b")
    st.sidebar.number_input("Approach Speed - CREEP B (mm/s)", value=st.session_state.speed_slow_b, step=10.0, key="speed_slow_b")
    st.sidebar.number_input("Acceleration Rate B (mm/s²)", value=st.session_state.accel_b, step=50.0, key="accel_b")
    st.sidebar.number_input("Deceleration Rate B (mm/s²)", value=st.session_state.decel_b, step=50.0, key="decel_b")
    st.sidebar.number_input("Slowdown Photoeye Offset B (mm)", value=st.session_state.sensor_distance_b, step=25.0, key="sensor_distance_b")
    st.sidebar.number_input("Drive Stop Ramp Time B (ms)", value=st.session_state.ramp_stop_b, step=10.0, min_value=0.0, key="ramp_stop_b")
    st.sidebar.number_input("Surface Friction Coefficient μ B", value=st.session_state.mu_b, step=0.01, min_value=0.01, max_value=1.0, key="mu_b")


# --- KINEMATIC & PHYSICAL COMPUTATION ENGINE ---
def simulate_kinematics(v_fast, v_slow, accel, decel, length, s_dist, ramp_stop_ms, mu):
    dt = 0.001  # 1 ms temporal resolution
    t_max = 30.0
    steps = int(t_max / dt)
    
    t = np.zeros(steps)
    pos = np.zeros(steps)
    vel = np.zeros(steps)
    
    p = 0.0
    v = 0.0
    pos_sensor_slowdown = length - s_dist
    pos_sensor_stop = length
    state = "ACCEL_FAST"
    
    t_sensor_slowdown = 0.0
    t_sensor_stop = 0.0
    
    # Mechanical compliance threshold (20 ms structural damping limit)
    T_MIN_MECHANICAL_MS = 20.0
    effective_ramp_stop_ms = max(ramp_stop_ms, T_MIN_MECHANICAL_MS)
    
    # Conveyor deceleration rate at stop position
    a_stop_conveyor = (v_slow / (effective_ramp_stop_ms / 1000.0))
    g_conveyor = (a_stop_conveyor / 1000.0) / 9.81
    
    # Maximum inertial friction limit before slip occurs
    g_max_payload = mu
    a_max_payload = mu * 9.81 * 1000.0  # mm/s²
    
    for i in range(1, steps):
        t[i] = t[i-1] + dt
        
        if state == "ACCEL_FAST":
            v += accel * dt
            if v >= v_fast:
                v = v_fast
                state = "CRUISE_FAST"
        elif state == "CRUISE_FAST":
            if p >= pos_sensor_slowdown:
                t_sensor_slowdown = t[i]
                state = "DECEL_TO_SLOW"
        elif state == "DECEL_TO_SLOW":
            v -= decel * dt
            if v <= v_slow:
                v = v_slow
                state = "CRUISE_SLOW"
        elif state == "CRUISE_SLOW":
            if p >= pos_sensor_stop:
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
        
        if state == "DONE" and i > 50 and np.all(vel[i-20:i] == 0):
            t = t[:i+1]
            pos = pos[:i+1]
            vel = vel[:i+1]
            break
            
    # Overtravel distance beyond the stop photoeye
    conveyor_overtravel_mm = pos[-1] - pos_sensor_stop
    
    # Payload dynamic slippage evaluation
    is_slippage_detected = g_conveyor > g_max_payload
    if is_slippage_detected:
        d_stopping_payload = (v_slow ** 2) / (2.0 * a_max_payload)
        d_stopping_conveyor = (v_slow ** 2) / (2.0 * a_stop_conveyor)
        relative_slip_mm = d_stopping_payload - d_stopping_conveyor
        effective_payload_g = g_max_payload
    else:
        relative_slip_mm = 0.0
        effective_payload_g = g_conveyor
        
    return t, pos, vel, t_sensor_slowdown, t_sensor_stop, conveyor_overtravel_mm, g_conveyor, effective_payload_g, is_slippage_detected, relative_slip_mm


# --- RUN SIMULATION ---
t_a, pos_a, vel_a, t_slow_a, t_stop_a, overtravel_a, g_conv_a, g_part_a, slip_a, d_slip_a = simulate_kinematics(
    st.session_state.speed_fast_a, st.session_state.speed_slow_a, 
    st.session_state.accel_a, st.session_state.decel_a, 
    st.session_state.conveyor_length, st.session_state.sensor_distance_a,
    st.session_state.ramp_stop_a, st.session_state.mu_a
)

if st.session_state.enable_comparison:
    t_b, pos_b, vel_b, t_slow_b, t_stop_b, overtravel_b, g_conv_b, g_part_b, slip_b, d_slip_b = simulate_kinematics(
        st.session_state.speed_fast_b, st.session_state.speed_slow_b, 
        st.session_state.accel_b, st.session_state.decel_b, 
        st.session_state.conveyor_length, st.session_state.sensor_distance_b,
        st.session_state.ramp_stop_b, st.session_state.mu_b
    )

# --- GRAPHICAL PLOTS ---
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Velocity Profile (mm/s)", "Position Trajectory (mm)"),
    horizontal_spacing=0.10
)

# Plot 1: Velocity
fig.add_trace(go.Scatter(x=t_a, y=vel_a, mode='lines', name='Velocity Profile A', line=dict(color='#1f77b4', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[t_slow_a, t_slow_a], y=[0, max(vel_a)*1.1], mode='lines', name='Slowdown PE Trigger A', line=dict(color="#ff7f0e", width=2, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=[t_stop_a, t_stop_a], y=[0, max(vel_a)*1.1], mode='lines', name='Stop PE Trigger A', line=dict(color="#d62728", width=2, dash="dash")), row=1, col=1)

if st.session_state.enable_comparison:
    fig.add_trace(go.Scatter(x=t_b, y=vel_b, mode='lines', name='Velocity Profile B', line=dict(color='#9467bd', width=3, dash='dashdot')), row=1, col=1)

# Plot 2: Position
fig.add_trace(go.Scatter(x=t_a, y=pos_a, mode='lines', name='Conveyor Position A', line=dict(color='#2ca02c', width=3)), row=1, col=2)
fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=st.session_state.conveyor_length, y1=st.session_state.conveyor_length, line=dict(color="#d62728", width=2, dash="dash"), row=1, col=2)
fig.add_shape(type="line", x0=0, x1=t_a[-1], y0=st.session_state.conveyor_length-st.session_state.sensor_distance_a, y1=st.session_state.conveyor_length-st.session_state.sensor_distance_a, line=dict(color="#ff7f0e", width=2, dash="dot"), row=1, col=2)

if st.session_state.enable_comparison:
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

# --- METRIC DASHBOARD ---
st.markdown("---")
st.subheader("📊 Dynamic Payload Stability & Inertial Slip Analysis (Profile A)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Conveyor Deceleration Rate", f"{g_conv_a:.3f} G")
col2.metric("Friction Threshold Limit (μ)", f"{st.session_state.mu_a:.2f} G")
col3.metric("Payload Slip Status", "🔴 SLIP DETECTED" if slip_a else "🟢 STABLE LOAD")
col4.metric("Relative Displacement (Slip)", f"{d_slip_a:.2f} mm" if slip_a else "0.00 mm")

st.markdown("---")
st.subheader("🎯 Conveyor Positioning & Cycle Dynamics (Profile A)")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Conveyor Overtravel", f"{overtravel_a:.2f} mm")
col6.metric("Final Conveyor Position", f"{pos_a[-1]:.2f} mm")
col7.metric("Final Payload Position", f"{(pos_a[-1] + d_slip_a):.2f} mm")
col8.metric("Total Motion Cycle Time", f"{t_a[-1]:.2f} s")

# Comparison Section
if st.session_state.enable_comparison:
    st.markdown("---")
    st.subheader("⚖️ Profile A vs Profile B Comparison Breakdown")
    
    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    c_b1.metric("Deceleration Rate B", f"{g_conv_b:.3f} G", delta=f"{(g_conv_b - g_conv_a):.3f} G", delta_color="inverse")
    c_b2.metric("Relative Slip B", f"{d_slip_b:.2f} mm", delta=f"{(d_slip_b - d_slip_a):.2f} mm", delta_color="inverse")
    c_b3.metric("Final Payload Position B", f"{(pos_b[-1] + d_slip_b):.2f} mm")
    c_b4.metric("Cycle Time Variance", f"{t_b[-1]:.2f} s", delta=f"{(t_b[-1] - t_a[-1]):.2f} s", delta_color="inverse")
