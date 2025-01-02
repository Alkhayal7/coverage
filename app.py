import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
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

# Helper functions for RSRP calculations
def calculate_rsrp_sa(d):
    """Calculate stochastic average RSRP"""
    return 33 * (-2.166 - np.log10(d + 1e-10))

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

def smooth_data(data, window_length=11, polyorder=3):
    """Smooth data using Savitzky-Golay filter"""
    return signal.savgol_filter(data, window_length, polyorder)

# Create tabs
tab1, tab2 = st.tabs(["RSRP Analysis", "Coverage Map"])

with tab1:
    st.header("RSRP Analysis")
    
    # Display equation
    st.latex(r"RSRP_{SA}(d) = 33 \times (-2.166 - \log_{10}(d))")
    
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
    
    # Time evolution controls
    col1, col2 = st.columns(2)
    with col1:
        time_steps = st.slider("Time Steps", 10, 200, 50)
    with col2:
        smoothing_window = st.slider("Smoothing Window", 3, 21, 11, step=2)
    
    # Calculate time evolution
    initial_rsrp = calculate_instantaneous_rsrp(track_distance)
    rsrp_evolution = evolve_rsrp(initial_rsrp, time_steps)
    rsrp_smooth = smooth_data(rsrp_evolution, smoothing_window)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(time_steps)),
        y=rsrp_evolution,
        mode='lines',
        name='Raw RSRP',
        line=dict(color='lightblue', width=1)
    ))
    fig2.add_trace(go.Scatter(
        x=list(range(time_steps)),
        y=rsrp_smooth,
        mode='lines',
        name='Smoothed RSRP',
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
    
    # Generate 3D surface data
    X, T = np.meshgrid(distances, np.arange(time_steps))
    Z = np.zeros_like(X)
    
    for i, d in enumerate(distances):
        init_rsrp = calculate_instantaneous_rsrp(d)
        Z[:, i] = evolve_rsrp(init_rsrp, time_steps)
    
    fig3 = go.Figure(data=[go.Surface(x=X, y=T, z=Z)])
    
    # Add tracked point as a scatter3d point
    fig3.add_trace(go.Scatter3d(
        x=[track_distance],
        y=[time_steps//2],  # Place in middle of time axis
        z=[Z[time_steps//2, np.argmin(np.abs(distances - track_distance))]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Tracked Point'
    ))
    
    fig3.update_layout(
        title='RSRP Evolution Over Distance and Time',
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
    
    # Upload floor plan
    uploaded_file = st.file_uploader("Upload Floor Plan", type=['png', 'jpg'])
    
    if uploaded_file is not None:
        # Load and display the floor plan
        image = Image.open(uploaded_file)
        width, height = image.size
        
        # Coverage map parameters
        col1, col2 = st.columns(2)
        with col1:
            gnb_x = st.slider("gNB X Position", 0, width-1, width//2)
            rsrp_min = st.number_input("Min RSRP (dB)", -140.0, -40.0, -120.0)
        with col2:
            gnb_y = st.slider("gNB Y Position", 0, height-1, height//2)
            rsrp_max = st.number_input("Max RSRP (dB)", -120.0, -20.0, -60.0)
        
        # Generate coverage map
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances from gNB
        distances = np.sqrt((X - gnb_x)**2 + (Y - gnb_y)**2)
        
        # Calculate RSRP values
        coverage = calculate_rsrp_sa(distances)
        
        # Create coverage map plot
        fig4 = go.Figure()
        
        # Add floor plan as background
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        
        fig4.add_trace(go.Image(
            z=image,
            opacity=0.5
        ))
        
        fig4.add_trace(go.Heatmap(
            z=coverage,
            colorscale='Viridis',
            zmin=rsrp_min,
            zmax=rsrp_max,
            opacity=0.7,
            colorbar=dict(title='RSRP (dB)')
        ))
        
        # Mark gNB location
        fig4.add_trace(go.Scatter(
            x=[gnb_x],
            y=[gnb_y],
            mode='markers',
            marker=dict(size=10, color='white', symbol='x'),
            name='gNB'
        ))
        
        fig4.update_layout(
            title='RSRP Coverage Map',
            width=width,
            height=height,
            showlegend=True
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Add threshold analysis
        st.subheader("Coverage Analysis")
        threshold = st.slider("RSRP Threshold (dB)", rsrp_min, rsrp_max, -100.0)
        
        below_threshold = np.sum(coverage < threshold)
        total_area = width * height
        coverage_percentage = 100 * (1 - below_threshold/total_area)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Coverage Percentage", f"{coverage_percentage:.1f}%")
        with col2:
            st.metric("Area Below Threshold", f"{below_threshold/total_area*100:.1f}%")