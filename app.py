import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw
import io

# Set page configuration for full width
st.set_page_config(layout="wide", page_title="RSRP Analysis Dashboard")

# Add custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .block-container { max-width: 100%; padding: 2rem; }
    </style>
""", unsafe_allow_html=True)


def calculate_rsrp_sa(d):
    """Calculate stochastic average RSRP using free space path loss model
    
    Parameters:
    d : float or numpy.ndarray
        Distance in meters
        
    Constants:
    - α (path loss exponent) = 2.9
    - f (frequency) = 3.5 GHz
    - Pt (transmit power) = 30 dBm
    """
    # Constants
    ALPHA = 2.9  # path loss exponent
    F = 3.5e9    # frequency in Hz
    PT = 30      # transmit power in dBm
    C = 3e8      # speed of light in m/s
    
    # Handle potential zero distance
    d = np.maximum(d, 1e-10)
    
    # Calculate RSRP
    return 10 * ALPHA * np.log10(C / (4 * np.pi * F * d)) + PT

def calculate_instantaneous_rsrp(d):
    """Calculate instantaneous RSRP with random variation"""
    rsrp_sa = calculate_rsrp_sa(d)
    v = np.random.uniform(-3, 1) * np.log10(d + 1e-10)
    return rsrp_sa + v

def evolve_rsrp(initial_rsrp, steps):
    """Evolve RSRP over time"""
    rsrp = [initial_rsrp]
    for _ in range(steps - 1):
        w = np.random.uniform(-1.5, 1.5)
        rsrp.append(rsrp[-1] + w)
    return np.array(rsrp)

# def evolve_rsrp(initial_rsrp, steps):
#     """Evolve RSRP over time with mean reversion
    
#     This implementation uses an Ornstein-Uhlenbeck-like process where values
#     tend to revert to their mean, creating more stable long-term behavior
#     """
#     rsrp = np.zeros(steps)
#     rsrp[0] = initial_rsrp
    
#     # Parameters
#     mean_reversion_strength = 0.3  # How strongly values return to the mean
#     noise_scale = 1.0  # Scale of random fluctuations
    
#     for i in range(1, steps):
#         # Calculate mean reversion term
#         deviation_from_initial = rsrp[i-1] - initial_rsrp
#         mean_reversion = -mean_reversion_strength * deviation_from_initial
        
#         # Add random noise with controlled scale
#         noise = np.random.uniform(-1, 1) * noise_scale
        
#         # Combine mean reversion and noise
#         rsrp[i] = rsrp[i-1] + mean_reversion + noise
    
#     return rsrp

def smooth_data(data, window_length=11, polyorder=3, method="moving average"):
    """Smooth data using Savitzky-Golay filter"""
    
    if method == "moving average":
        return np.convolve(data, np.ones(window_length)/window_length, mode='valid')
    if method == "gaussian filter":
        return gaussian_filter1d(data, sigma=window_length)
    if method == "savgol filter":
        return signal.savgol_filter(data, window_length, polyorder)

# Create tabs
tab1, tab2 = st.tabs(["RSRP Analysis", "Coverage Map"])

with tab1:
    st.header("RSRP Analysis")
    
    # Display equation
    st.latex(r"RSRP_{SA}(d) = 10\alpha \log_{10}\left(\frac{c}{4\pi fd}\right) + P_t")

    # Add parameters explanation
    st.markdown("""
    Where:
    - α = 2.9 (path loss exponent)
    - f = 3.5 GHz (frequency)
    - c = 3×10⁸ m/s (speed of light)
    - Pt = 30 dBm (transmit power)
    """)
    
    # Basic RSRP vs Distance parameters
    d_max = st.slider("Maximum Distance (m)", 10, 1000, 500)
    track_distance = st.slider("Track Distance (m)", 1, d_max, d_max//2)
    
    # Generate distance array and RSRP values
    distances = np.linspace(1, d_max, 500)  # Fixed 500 points for smooth curve
    rsrp_sa = calculate_rsrp_sa(distances)
    
    # Get tracked point values
    tracked_rsrp = calculate_rsrp_sa(track_distance)
    
    # Create RSRP vs Distance plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=distances,
        y=rsrp_sa,
        mode='lines',
        name='RSRP_SA(d)',
        line=dict(color='blue', width=2)
    ))
    
    # Add tracked point
    fig1.add_trace(go.Scatter(
        x=[track_distance],
        y=[tracked_rsrp],
        mode='markers',
        name='Tracked Point',
        marker=dict(color='red', size=10)
    ))
    
    # Add reference lines
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
    
    # Display tracked values
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Distance", f"{track_distance:.1f} m")
    with col2:
        st.metric("RSRP Value", f"{tracked_rsrp:.2f} dB")
    
    # Time evolution section
    st.subheader("Temporal Evolution")
    
    # Initialize random state if not already done
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = np.random.randint(0, 1000000)
        
    if 'initial_rsrp' not in st.session_state:
        # Set random seed for initial RSRP calculation
        np.random.seed(st.session_state.random_seed)
        st.session_state.initial_rsrp = calculate_instantaneous_rsrp(track_distance)
    
    # Time evolution controls
    col1, col2, col3 = st.columns(3)
    with col1:
        time_steps = st.slider("Time Steps", 10, 500, 100)
    
    # Calculate time evolution with consistent random state
    if 'rsrp_evolution' not in st.session_state or len(st.session_state.rsrp_evolution) != time_steps:
        # Set random seed before evolution
        np.random.seed(st.session_state.random_seed)
        st.session_state.rsrp_evolution = evolve_rsrp(st.session_state.initial_rsrp, time_steps)
    
    # Smoothing controls
    with col2:
        smoothing_method = st.selectbox(
            "Smoothing Method",
            ["gaussian filter", "moving average", "savgol filter"]
        )
    
    with col3:
        if smoothing_method == "savgol filter":
            smoothing_window = st.slider("Smoothing Window", 3, 21, 9, step=2)
        else:
            smoothing_window = st.slider("Smoothing Window", 1, 21, 2, step=1)
    
    # Apply smoothing to the stored evolution data
    rsrp_smooth = smooth_data(st.session_state.rsrp_evolution, smoothing_window, method=smoothing_method)
    
    # Reset button to generate new random data
    if st.button("Generate New Random Data"):
        st.session_state.random_seed = np.random.randint(0, 1000000)
        del st.session_state.initial_rsrp
        del st.session_state.rsrp_evolution
        st.rerun()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(time_steps)),
        y=st.session_state.rsrp_evolution,
        mode='lines',
        name='Raw RSRP',
        line=dict(color='lightblue', width=1)
    ))
    fig2.add_trace(go.Scatter(
        x=list(range(len(rsrp_smooth))),
        y=rsrp_smooth,
        mode='lines',
        name=f'Smoothed RSRP ({smoothing_method})',
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
    
    # Add option for smoothed view
    show_smoothed = st.checkbox("Show Smoothed RSRP in 3D", value=False)
    
    # Generate 3D surface data
    X, T = np.meshgrid(distances, np.arange(time_steps))
    Z = np.zeros_like(X)
    
    for i, d in enumerate(distances):
        init_rsrp = calculate_instantaneous_rsrp(d)
        raw_evolution = evolve_rsrp(init_rsrp, time_steps)
        
        if show_smoothed:
            # Apply the same smoothing as selected above
            Z[:, i] = smooth_data(raw_evolution, smoothing_window, method=smoothing_method)
        else:
            Z[:, i] = raw_evolution
    
    fig3 = go.Figure(data=[go.Surface(
        x=X, 
        y=T, 
        z=Z,
        # colorscale='Viridis',
        name='RSRP Evolution'
    )])
    
    # Add tracked point as a scatter3d point
    tracked_z = Z[time_steps//2, np.argmin(np.abs(distances - track_distance))]
    fig3.add_trace(go.Scatter3d(
        x=[track_distance],
        y=[time_steps//2],  # Place in middle of time axis
        z=[tracked_z],
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
    
    # Load the default floor plan
    default_map_path = "/home/mobisense/Desktop/coverage/maps/original.png"
    image = Image.open(default_map_path)
    width, height = image.size
    
    # Add scaling options
    st.subheader("Map Scaling")
    col1, col2 = st.columns(2)
    with col1:
        real_width = st.number_input("Real Width (meters)", min_value=1.0, value=250.0)
    with col2:
        real_height = st.number_input("Real length (meters)", min_value=1.0, value=187.5)
    
    # Calculate scaling factors
    meters_per_pixel_x = real_width / width
    meters_per_pixel_y = real_height / height
    
    # Coverage map parameters with real-world coordinates
    col1, col2 = st.columns(2)
    with col1:
        gnb_x_meters = st.number_input("gNB X Position (meters)", 
                                     0.0, real_width, 
                                     20.0)
        gnb_y_meters = st.number_input("gNB Y Position (meters)", 
                                     0.0, real_height, 
                                     31.0)
    with col2:
        rsrp_min = st.number_input("Min RSRP (dB)", -140.0, -40.0, -120.0)
        rsrp_max = st.number_input("Max RSRP (dB)", -120.0, -20.0, -60.0)
    
    # Convert gNB position from meters to pixels
    gnb_x = int(gnb_x_meters / meters_per_pixel_x)
    gnb_y = int(gnb_y_meters / meters_per_pixel_y)
    
    # Generate base distance grid in meters
    x_pixels = np.arange(width)
    y_pixels = np.arange(height)
    X_pixels, Y_pixels = np.meshgrid(x_pixels, y_pixels)
    
    # Calculate distances in meters
    distances = np.sqrt(
        ((X_pixels - gnb_x) * meters_per_pixel_x)**2 + 
        ((Y_pixels - gnb_y) * meters_per_pixel_y)**2
    )
    
    # Constants
    ALPHA = 2.9  # path loss exponent
    F = 3.5e9    # frequency in Hz
    PT = 30      # transmit power in dBm
    C = 3e8      # speed of light in m/s

    # Calculate RSRP with the real distances
    coverage = np.zeros_like(distances)
    np.random.seed(42)

    for i in range(width):
        for j in range(height):
            d = distances[j, i]
            # Handle potential zero distance
            d = np.maximum(d, 1e-10)
            
            # Add random variation to model shadowing/fading
            v = np.random.uniform(-3, 1)
            
            # Calculate RSRP using the new formula
            coverage[j, i] = 10 * ALPHA * np.log10(C / (4 * np.pi * F * d)) + PT + v
    
    # Create coverage map plot
    fig4 = go.Figure()
    
    # Add floor plan
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    fig4.add_trace(go.Image(z=image, opacity=0.5))
    
    # Calculate positions in meters
    X_meters = X_pixels * meters_per_pixel_x
    Y_meters = Y_pixels * meters_per_pixel_y

    # Stack the custom data into a 3D array where each pixel has [distance, x_meters, y_meters]
    hover_data = np.dstack((distances, X_meters, Y_meters))

    # Add heatmap
    fig4.add_trace(go.Heatmap(
        z=coverage,
        colorscale='RdBu_r',
        zmin=rsrp_min,
        zmax=rsrp_max,
        opacity=0.7,
        colorbar=dict(title='RSRP (dB)'),
        # Add custom hover template with x,y positions
        hovertemplate='RSRP: %{z:.1f} dB<br>' +
                    'Distance from gNB: %{customdata[0]:.1f}m<br>' +
                    'X: %{customdata[1]:.1f}m<br>' +
                    'Y: %{customdata[2]:.1f}m' +
                    '<extra></extra>',
        # Include distances and positions as custom data
        customdata=hover_data
    ))
        
    # Mark gNB location
    fig4.add_trace(go.Scatter(
        x=[gnb_x],
        y=[gnb_y],
        mode='markers',
        marker=dict(size=10, color='white', symbol='x'),
        name='gNB'
    ))
    
    # Add scale bar (50 pixels = 5 meters)
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