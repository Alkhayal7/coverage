import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from PIL import Image
import io

# Cache heavy computations
@st.cache_data
def load_and_process_image(image_path):
    """Load and process the floor plan image"""
    return Image.open(image_path)

@st.cache_data
def calculate_rsrp_sa(d):
    """Vectorized RSRP calculation"""
    ALPHA, F, PT = 2.9, 3.5e9, 30
    C = 3e8
    d = np.maximum(d, 1e-10)
    return 10 * ALPHA * np.log10(C / (4 * np.pi * F * d)) + PT

@st.cache_data
def calculate_coverage_map(width, height, gnb_x, gnb_y, meters_per_pixel_x, meters_per_pixel_y):
    """Vectorized coverage map calculation"""
    x_pixels = np.arange(width)
    y_pixels = np.arange(height)
    X_pixels, Y_pixels = np.meshgrid(x_pixels, y_pixels)
    
    # Calculate distances in meters (vectorized)
    distances = np.sqrt(
        ((X_pixels - gnb_x) * meters_per_pixel_x)**2 + 
        ((Y_pixels - gnb_y) * meters_per_pixel_y)**2
    )
    
    # Calculate base RSRP
    coverage = calculate_rsrp_sa(distances)
    
    # Add random variation (vectorized)
    np.random.seed(42)
    variation = np.random.uniform(-3, 1, size=distances.shape)
    coverage += variation
    
    # Calculate position matrices for hover data
    X_meters = X_pixels * meters_per_pixel_x
    Y_meters = Y_pixels * meters_per_pixel_y
    
    return coverage, distances, X_meters, Y_meters

def calculate_instantaneous_rsrp(d):
    """Calculate instantaneous RSRP with random variation"""
    rsrp_sa = calculate_rsrp_sa(d)
    v = np.random.uniform(-3, 1) * np.log10(d + 1e-10)
    return rsrp_sa + v

def evolve_rsrp(initial_rsrp, steps):
    """Evolve RSRP over time"""
    rsrp = np.zeros(steps)
    rsrp[0] = initial_rsrp
    w = np.random.uniform(-1.5, 1.5, size=steps-1)
    rsrp[1:] = rsrp[0] + np.cumsum(w)
    return rsrp

@st.cache_data
def smooth_data(data, window_length=11, polyorder=3, method="moving average"):
    """Smoothed data calculation with caching"""
    if method == "moving average":
        kernel = np.ones(window_length)/window_length
        return np.convolve(data, kernel, mode='valid')
    if method == "gaussian filter":
        return gaussian_filter1d(data, sigma=window_length)
    return signal.savgol_filter(data, window_length, polyorder)

# Set page configuration
st.set_page_config(layout="wide", page_title="RSRP Analysis Dashboard")

# Add custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .block-container { max-width: 100%; padding: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["RSRP Analysis", "Coverage Map"])

with tab1:
    st.header("RSRP Analysis")
    
    # Display equation and parameters
    st.latex(r"RSRP_{SA}(d) = 10\alpha \log_{10}\left(\frac{c}{4\pi fd}\right) + P_t")
    st.markdown("""
    Where:
    - α = 2.9 (path loss exponent)
    - f = 3.5 GHz (frequency)
    - c = 3×10⁸ m/s (speed of light)
    - Pt = 30 dBm (transmit power)
    """)
    
    # RSRP vs Distance analysis
    d_max = st.slider("Maximum Distance (m)", 10, 1000, 500)
    track_distance = st.slider("Track Distance (m)", 1, d_max, d_max//2)
    
    # Generate distance array and RSRP values
    distances = np.linspace(1, d_max, 500)
    rsrp_sa = calculate_rsrp_sa(distances)
    tracked_rsrp = calculate_rsrp_sa(track_distance)
    
    # Create RSRP vs Distance plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=distances, y=rsrp_sa,
        mode='lines', name='RSRP_SA(d)',
        line=dict(color='blue', width=2)
    ))
    
    fig1.add_trace(go.Scatter(
        x=[track_distance], y=[tracked_rsrp],
        mode='markers', name='Tracked Point',
        marker=dict(color='red', size=10)
    ))
    
    for level in [-80, -110]:
        fig1.add_hline(y=level, line_dash="dash", line_color="gray",
                      annotation_text=f"{level} dB")
    
    fig1.update_layout(
        title="RSRP vs Distance",
        xaxis_title="Distance (m)",
        yaxis_title="RSRP (dB)",
        height=500
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Distance", f"{track_distance:.1f} m")
    with col2:
        st.metric("RSRP Value", f"{tracked_rsrp:.2f} dB")
    
    # Time evolution section
    st.subheader("Temporal Evolution")
    
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = np.random.randint(0, 1000000)
        
    if 'initial_rsrp' not in st.session_state:
        np.random.seed(st.session_state.random_seed)
        st.session_state.initial_rsrp = calculate_instantaneous_rsrp(track_distance)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        time_steps = st.slider("Time Steps", 10, 500, 100)
    
    if 'rsrp_evolution' not in st.session_state or len(st.session_state.rsrp_evolution) != time_steps:
        np.random.seed(st.session_state.random_seed)
        st.session_state.rsrp_evolution = evolve_rsrp(st.session_state.initial_rsrp, time_steps)
    
    with col2:
        smoothing_method = st.selectbox(
            "Smoothing Method",
            ["gaussian filter", "moving average", "savgol filter"]
        )
    
    with col3:
        smoothing_window = st.slider(
            "Smoothing Window",
            3 if smoothing_method == "savgol filter" else 1,
            21,
            9 if smoothing_method == "savgol filter" else 2,
            step=2 if smoothing_method == "savgol filter" else 1
        )
    
    rsrp_smooth = smooth_data(st.session_state.rsrp_evolution, smoothing_window, method=smoothing_method)
    
    if st.button("Generate New Random Data"):
        st.session_state.random_seed = np.random.randint(0, 1000000)
        del st.session_state.initial_rsrp
        del st.session_state.rsrp_evolution
        st.rerun()
    
    # Time evolution plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=np.arange(time_steps), y=st.session_state.rsrp_evolution,
        mode='lines', name='Raw RSRP',
        line=dict(color='lightblue', width=1)
    ))
    
    fig2.add_trace(go.Scatter(
        x=np.arange(len(rsrp_smooth)), y=rsrp_smooth,
        mode='lines', name=f'Smoothed RSRP ({smoothing_method})',
        line=dict(color='blue', width=2)
    ))
    
    fig2.update_layout(
        title=f"RSRP Time Evolution at d={track_distance}m",
        xaxis_title="Time Step",
        yaxis_title="RSRP (dB)",
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3D visualization
    st.subheader("3D RSRP Evolution")
    show_smoothed = st.checkbox("Show Smoothed RSRP in 3D", value=True)
    
    @st.cache_data
    def generate_3d_surface(distances, time_steps, smoothing_window, smoothing_method, show_smoothed):
        X, T = np.meshgrid(distances, np.arange(time_steps))
        Z = np.zeros_like(X)
        
        for i, d in enumerate(distances):
            init_rsrp = calculate_instantaneous_rsrp(d)
            raw_evolution = evolve_rsrp(init_rsrp, time_steps)
            Z[:, i] = smooth_data(raw_evolution, smoothing_window, method=smoothing_method) if show_smoothed else raw_evolution
            
        return X, T, Z
    
    X, T, Z = generate_3d_surface(distances, time_steps, smoothing_window, smoothing_method, show_smoothed)
    
    fig3 = go.Figure(data=[go.Surface(x=X, y=T, z=Z)])
    
    tracked_z = Z[time_steps//2, np.argmin(np.abs(distances - track_distance))]
    fig3.add_trace(go.Scatter3d(
        x=[track_distance], y=[time_steps//2], z=[tracked_z],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Tracked Point'
    ))
    
    fig3.update_layout(
        title=f'RSRP Evolution Over Distance and Time ({("Smoothed" if show_smoothed else "Raw")})',
        scene=dict(
            xaxis_title='Distance (m)',
            yaxis_title='Time Step',
            zaxis_title='RSRP (dB)'
        ),
        height=700
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.header("Coverage Map")
    
    # Load and cache the floor plan
    default_map_path = "maps/original.png"
    image = load_and_process_image(default_map_path)
    width, height = image.size
    
    # Map scaling
    st.subheader("Map Scaling")
    col1, col2 = st.columns(2)
    with col1:
        real_width = st.number_input("Real Width (meters)", min_value=1.0, value=250.0)
    with col2:
        real_height = st.number_input("Real length (meters)", min_value=1.0, value=187.5)
    
    meters_per_pixel_x = real_width / width
    meters_per_pixel_y = real_height / height
    
    # Coverage parameters
    col1, col2 = st.columns(2)
    with col1:
        gnb_x_meters = st.number_input("gNB X Position (meters)", 0.0, real_width, 20.0)
        gnb_y_meters = st.number_input("gNB Y Position (meters)", 0.0, real_height, 31.0)
    with col2:
        rsrp_min = st.number_input("Min RSRP (dB)", -140.0, -40.0, -120.0)
        rsrp_max = st.number_input("Max RSRP (dB)", -120.0, -20.0, -60.0)
    
    # Convert gNB position
    gnb_x = int(gnb_x_meters / meters_per_pixel_x)
    gnb_y = int(gnb_y_meters / meters_per_pixel_y)
    
    # Calculate coverage map
    coverage, distances, X_meters, Y_meters = calculate_coverage_map(
        width, height, gnb_x, gnb_y, 
        meters_per_pixel_x, meters_per_pixel_y
    )
    
    # Create coverage map plot
    fig4 = go.Figure()
    
    # Add floor plan
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    fig4.add_trace(go.Image(z=image, opacity=0.5))
    
    # Add heatmap with hover data
    hover_data = np.dstack((distances, X_meters, Y_meters))
    fig4.add_trace(go.Heatmap(
        z=coverage,
        colorscale='RdBu_r',
        zmin=rsrp_min,
        zmax=rsrp_max,
        opacity=0.7,
        colorbar=dict(title='RSRP (dB)'),
        hovertemplate='RSRP: %{z:.1f} dB<br>' +
                    'Distance from gNB: %{customdata[0]:.1f}m<br>' +
                    'X: %{customdata[1]:.1f}m<br>' +
                    'Y: %{customdata[2]:.1f}m' +
                    '<extra></extra>',
        customdata=hover_data
    ))
    
    # Add gNB marker
    fig4.add_trace(go.Scatter(
        x=[gnb_x], y=[gnb_y],
        mode='markers',
        marker=dict(size=10, color='white', symbol='x'),
        name='gNB'
    ))
    
    # Add scale bar
    scale_bar_meters = 5
    scale_bar_pixels = int(scale_bar_meters / meters_per_pixel_x)
    fig4.add_trace(go.Scatter(
        x=[width-scale_bar_pixels-10, width-10],
        y=[height-20, height-20],
        mode='lines',
        line=dict(color='white', width=3),
        name=f'{scale_bar_meters}m'
    ))
    
    fig4.update_layout(
        title='RSRP Coverage Map',
        width=width,
        height=height,
        showlegend=True
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Display map information
    st.subheader("Map Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Map Width", f"{real_width:.1f} meters")
        st.metric("Map Height", f"{real_height:.1f} meters")
    with col2:
        st.metric("Resolution", f"{meters_per_pixel_x:.2f} meters/pixel")
        st.metric("gNB Position", f"({gnb_x_meters:.1f}m, {gnb_y_meters:.1f}m)")
    
    # Display equations
    st.subheader("Coverage Calculations")
    st.latex(r"d(x,y) = \sqrt{(x - x_{gNB})^2 + (y - y_{gNB})^2}")
    st.latex(r"RSRP(x,y) = 10\alpha \log_{10}\left(\frac{c}{4\pi fd(x,y)}\right) + P_t + v")
    
    # Add parameters explanation
    st.markdown("""
    Where:
    - d(x,y) is the distance from any point to the gNB
    - α = 2.9 (path loss exponent)
    - f = 3.5 GHz (frequency)
    - c = 3×10⁸ m/s (speed of light)
    - Pt = 30 dBm (transmit power)
    - v is random variation ∈ [-3,1]
    """)