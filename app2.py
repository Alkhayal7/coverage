import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D  # Ensures 3D plotting is recognized

# ----------------------------
# 1. RSRP Calculation Helpers
# ----------------------------

def calculate_rsrp(d: float) -> float:
    """
    Calculate the Stochastic Average (SA) RSRP at distance d:
    RSRP_SA(d) = 33 * ( -2.166 - log10(d) ).
    """
    if d <= 0:
        return None
    return 33 * (-2.166 - np.log10(d))


def calculate_instantaneous_rsrp(d: float) -> float:
    """
    Instantaneous RSRP at t=0, adding a stochastic term:
    RSRP(d,t_0) = RSRP_SA(d) + v,
    where v ~ Uniform([-3, 1]) * log10(d).
    """
    rsrp_sa = calculate_rsrp(d)
    if rsrp_sa is None:
        return None
    v = np.random.uniform(-3, 1) * np.log10(d)
    return rsrp_sa + v


def evolve_rsrp_over_time(d: float, steps: int) -> np.ndarray:
    """
    Temporal evolution of RSRP over 'steps' time instances:
      RSRP(d, t_i) = RSRP(d, t_{i-1}) + w,
    where w ~ Uniform([-1.5, 1.5]).
    """
    rsrp_values = [calculate_instantaneous_rsrp(d)]
    for _ in range(1, steps):
        w = np.random.uniform(-1.5, 1.5)
        rsrp_values.append(rsrp_values[-1] + w)
    return np.array(rsrp_values)


def smooth_curve(data: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    """
    Smooth the RSRP curve using Savitzky-Golay filter.
    Increase 'window' or 'poly' for stronger smoothing.
    """
    if len(data) < window:
        # If data is shorter than the smoothing window, just return the raw data
        return data
    return savgol_filter(data, window_length=window, polyorder=poly)


# ----------------------------
# 2. Coverage Map Helpers
# ----------------------------

def rsrp_sa_coverage(distance_m: float, random_variation: float = 0.0) -> float:
    """
    Computes RSRP_SA(d) = 33 * (-2.166 - log10(d)) + v
    distance_m: distance in meters
    random_variation: optional small variation (v)
    """
    if distance_m <= 0:
        return np.nan
    return 33 * (-2.166 - np.log10(distance_m)) + random_variation


def generate_polygon_mask(polygon_points, img_width, img_height):
    """
    Create a boolean mask of the polygon area
    where polygon_points = [(x1, y1), (x2, y2), ...].
    """
    # Create a blank image for the mask
    mask_img = Image.new("L", (img_width, img_height), 0)
    # Draw the polygon on the mask
    ImageDraw.Draw(mask_img).polygon(polygon_points, outline=1, fill=1)
    # Convert to NumPy array of booleans
    mask = np.array(mask_img)
    polygon_mask = mask.astype(bool)
    return polygon_mask


def point_in_polygon(x, y, polygon_mask):
    """
    Checks if point (x, y) is inside polygon_mask.
    polygon_mask is assumed to be a 2D boolean array.
    """
    h, w = polygon_mask.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return polygon_mask[int(y), int(x)]


# ----------------------------
# 3. Streamlit App
# ----------------------------

# Set page configuration to 'centered' layout
st.set_page_config(page_title="RSRP & Coverage Map Visualization", layout="centered")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Set a light background color for the main content */
    .main {
        background-color: #f5f5f5;
    }
    /* Style the main title */
    .main > div > div > div > h1 {
        color: #333333;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Style the headers */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #333333;
        text-align: center;
    }
    /* Style the sliders and number inputs */
    .stSlider > div > div, .stNumberInput > div > div {
        color: #555555;
    }
    /* Style the download button */
    .stDownloadButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 12px;
        cursor: pointer;
    }
    /* Style the expander */
    .streamlit-expanderHeader {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Center the main title using HTML with styling
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="display: inline-block; background-color: #e6f7ff; padding: 20px 40px; border-radius: 15px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
        RSRP & Coverage Map Visualization
    </h1>
</div>
""", unsafe_allow_html=True)

# Create Tabs for better organization
tabs = st.tabs(["RSRP Pipeline Visualization", "Coverage Map Plot"])

# ----------------------------
# Tab 1: RSRP Pipeline Visualization
# ----------------------------
with tabs[0]:
    st.header("RSRP Pipeline Visualization")

    # ----------------------------
    # SECTION A: RSRP vs Distance (2D Plot)
    # ----------------------------
    st.subheader("1. RSRP vs. Distance")

    # Centered LaTeX Equation for RSRP_SA(d)
    st.latex(r"\text{RSRP}_{SA}(d) = 33 \times \left( -2.166 - \log_{10}(d) \right)")

    # Controls for Distance Plot
    col1, col2 = st.columns(2, gap="small")
    with col1:
        d_max = st.slider("Max Distance (d_max) [m]:", 100, 2000, 500, step=50, key="d_max_slider")
    with col2:
        d = st.slider("Specific Distance (d) [m]:", 1, d_max, 50, step=1, key="d_slider")

    distances = np.linspace(1, d_max, 500)
    rsrp_values = [calculate_rsrp(x) for x in distances]

    # Plot RSRP vs Distance
    fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
    ax_dist.plot(distances, rsrp_values, label="RSRP_SA(d)", color='#1f77b4', linewidth=2)
    ax_dist.axvline(d, color='#ff7f0e', linestyle='--', label=f"Selected d = {d} m")
    ax_dist.set_xlabel("Distance (m)", fontsize=12)
    ax_dist.set_ylabel("RSRP (dB)", fontsize=12)
    ax_dist.set_title("RSRP vs. Distance", fontsize=14)
    ax_dist.legend(fontsize=10)
    ax_dist.tick_params(axis='both', which='major', labelsize=10)
    ax_dist.grid(True, linestyle='--', alpha=0.5)

    # Calculate RSRP at selected distance
    rsrp_at_d = calculate_rsrp(d)
    if rsrp_at_d is not None:
        ax_dist.plot(d, rsrp_at_d, 'ro')  # Red dot at the selected point
        ax_dist.text(d, rsrp_at_d + 0.5, f"{rsrp_at_d:.2f} dB", color='red', fontsize=10, ha='left')

    # Add horizontal reference lines for typical RSRP levels
    typical_levels = [-80, -110]  # Example thresholds
    for level in typical_levels:
        ax_dist.axhline(level, color='grey', linestyle=':', linewidth=1)
        ax_dist.text(d_max, level + 0.5, f"{level} dB", color='grey', fontsize=10, ha='right')

    st.pyplot(fig_dist)

    # ----------------------------
    # SECTION B: Instantaneous RSRP & Time Evolution (2D Plot)
    # ----------------------------
    st.subheader("2. Instantaneous and Temporal Evolution at Selected Distance")

    # Centered LaTeX Equations for Instantaneous and Temporal Evolution
    st.latex(r"""
    \begin{aligned}
    \text{Instantaneous RSRP at } t=0:& \quad \text{RSRP}(d, t_0) = \text{RSRP}_{SA}(d) + v \\
    v \sim \mathcal{U}([-3, 1]) \times \log_{10}(d) \\
    \text{Temporal Evolution:}& \quad \text{RSRP}(d, t_i) = \text{RSRP}(d, t_{i-1}) + w \\
    w \sim \mathcal{U}([-1.5, 1.5])
    \end{aligned}
    """)

    # Controls for Time Evolution
    col3, col4, col5 = st.columns(3, gap="small")
    with col3:
        time_steps = st.slider("Number of Time Steps:", 10, 200, 50, step=10, key="time_steps_slider")
    with col4:
        window_size = st.slider("Savgol Window Length:", 3, 31, 21, step=2, key="window_size_slider")
    with col5:
        poly_order = st.slider("Savgol Polynomial Order:", 1, 5, 3, step=1, key="poly_order_slider")

    # Calculate instantaneous RSRP at t=0
    instantaneous_rsrp = calculate_instantaneous_rsrp(d)
    if instantaneous_rsrp is not None:
        st.markdown(f"**Instantaneous RSRP at d = {d} m (t=0):** {instantaneous_rsrp:.2f} dB")
    else:
        st.error("Invalid distance selected.")

    # Evolve RSRP over time and smooth
    rsrp_time_series = evolve_rsrp_over_time(d, time_steps)
    smoothed_rsrp = smooth_curve(rsrp_time_series, window=window_size, poly=poly_order)

    # Plot the raw vs. smoothed time series
    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    ax_time.plot(np.arange(time_steps), rsrp_time_series, label="Raw RSRP", alpha=0.6, color='#2ca02c')
    ax_time.plot(np.arange(time_steps), smoothed_rsrp, label="Smoothed RSRP", color='#d62728', linewidth=2)
    ax_time.set_xlabel("Time Step", fontsize=12)
    ax_time.set_ylabel("RSRP (dB)", fontsize=12)
    ax_time.set_title(f"Temporal Evolution at d = {d} m", fontsize=14)
    ax_time.legend(fontsize=10)
    ax_time.tick_params(axis='both', which='major', labelsize=10)
    ax_time.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig_time)

    # ----------------------------
    # SECTION C: 3D Visualization Over Distance & Time
    # ----------------------------
    st.subheader("3. 3D Visualization (Distance vs. Time vs. RSRP)")

    # Centered LaTeX Equations for 3D RSRP Over Distance and Time
    st.latex(r"""
    \begin{aligned}
    \text{RSRP}(d, t_i) &= \text{RSRP}(d, t_{i-1}) + w \\
    w &\sim \mathcal{U}([-1.5, 1.5])
    \end{aligned}
    """)

    # Generate meshgrid for 3D plot
    time_array = np.arange(time_steps)  # 0, 1, 2, ... up to time_steps-1
    X, T = np.meshgrid(distances, time_array)

    # Initialize Z array
    Z = np.zeros((time_steps, len(distances)))
    for i, dist in enumerate(distances):
        # Evolve RSRP for each distance across the same time_steps
        Z[:, i] = evolve_rsrp_over_time(dist, time_steps)

    # Plot the 3D surface
    fig_3d = plt.figure(figsize=(10, 7))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    surf = ax_3d.plot_surface(X, T, Z, cmap='viridis', edgecolor='none')
    ax_3d.set_xlabel("Distance (m)", fontsize=12, labelpad=10)
    ax_3d.set_ylabel("Time Step", fontsize=12, labelpad=10)
    ax_3d.set_zlabel("RSRP (dB)", fontsize=12, labelpad=10)
    ax_3d.set_title("RSRP Over Distance and Time", fontsize=16)
    ax_3d.view_init(elev=30, azim=220)  # Adjust view angle for better visualization

    # Add colorbar
    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, pad=0.1, label="RSRP (dB)")

    st.pyplot(fig_3d)

    # ----------------------------
    # SECTION D: Download Smoothed Time Series
    # ----------------------------
    st.subheader("4. Download Smoothed Time Series Data")

    # Centered LaTeX Equation for Downloading Data
    st.latex(r"\text{Download the smoothed RSRP time series data as a CSV file.}")

    # Prepare CSV data
    csv_data = "TimeStep,RSRP(dB)\n"
    for i, val in enumerate(smoothed_rsrp):
        csv_data += f"{i},{val:.2f}\n"

    # Download button
    st.download_button(
        label="Download Smoothed RSRP Data as CSV",
        data=csv_data,
        file_name="rsrp_smoothed.csv",
        mime="text/csv",
    )

# ----------------------------
# Tab 2: Coverage Map Plot
# ----------------------------
with tabs[1]:
    st.header("Coverage Map Plot (RSRP)")

    # 3.1 Upload Floor Plan
    uploaded_file = st.file_uploader("Upload the Refinery Floor Plan", type=["jpg", "png"], key="coverage_map_upload")

    if uploaded_file is not None:
        # Load the floor plan as RGBA (to keep transparency if any)
        floor_plan_image = Image.open(uploaded_file).convert("RGBA")
        w, h = floor_plan_image.size

        st.markdown(f"**Floor Plan Dimensions:** {w} x {h} pixels")

        # 3.2 Define a polygon (the accessible area).
        #    For simplicity, we'll use the entire image as the polygon:
        #    You can modify polygon_points as needed or implement interactive polygon drawing
        polygon_points = [(0, 0), (w, 0), (w, h), (0, h)]
        polygon_mask = generate_polygon_mask(polygon_points, w, h)

        # 3.3 Let the user pick the gNB location (in pixel coordinates).
        st.subheader("Select gNB Location (Pixel Coordinates)")
        col1, col2 = st.columns(2, gap="small")
        with col1:
            gNB_x = st.slider("gNB X coordinate", 0, w - 1, w // 2, step=1, key="coverage_gNB_x")
        with col2:
            gNB_y = st.slider("gNB Y coordinate", 0, h - 1, h // 2, step=1, key="coverage_gNB_y")
        st.markdown(f"**Current gNB location:** (x={gNB_x}, y={gNB_y})")

        # 3.4 Configure Heatmap Min/Max
        st.subheader("Configure Heatmap")

        # Centered LaTeX Equation for Coverage Heatmap
        st.latex(r"\text{RSRP}(d, t) = 33 \times \left( -2.166 - \log_{10}(d) \right)")

        col3, col4 = st.columns(2, gap="small")
        with col3:
            rsrp_min = st.number_input("RSRP Min (dB)", value=-140.0, step=5.0, key="coverage_rsrp_min")
        with col4:
            rsrp_max = st.number_input("RSRP Max (dB)", value=-40.0, step=5.0, key="coverage_rsrp_max")

        # 3.5 Downsampling factor
        downsample_factor = st.slider("Downsample Factor", 1, 20, 4, step=1, key="coverage_downsample_factor")

        # Prepare an array to store coverage map values
        #   Rows = ceil(h / downsample_factor)
        #   Cols = ceil(w / downsample_factor)
        map_height = math.ceil(h / downsample_factor)
        map_width = math.ceil(w / downsample_factor)
        coverage_map = np.full((map_height, map_width), np.nan, dtype=np.float32)

        # 3.6 Compute Coverage for each (x, y) in the polygon, downsampled
        for yy in range(0, h, downsample_factor):
            for xx in range(0, w, downsample_factor):
                row = yy // downsample_factor
                col = xx // downsample_factor

                # Sanity check to avoid index errors
                if row >= map_height or col >= map_width:
                    continue  # skip any out-of-bounds

                # Check if inside polygon
                if point_in_polygon(xx, yy, polygon_mask):
                    # Distance from gNB in pixel space
                    dx = xx - gNB_x
                    dy = yy - gNB_y
                    distance_pixels = np.sqrt(dx**2 + dy**2)

                    # If your map has a known scale (e.g., 1 pixel = 0.2 m), multiply here:
                    distance_m = distance_pixels  # e.g., distance_pixels * 0.2

                    # RSRP with optional random variation
                    # For coverage map, use RSRP_SA without variation
                    rsrp_value = rsrp_sa_coverage(distance_m)
                    coverage_map[row, col] = rsrp_value
                else:
                    coverage_map[row, col] = np.nan  # outside polygon

        # 3.7 Plot coverage heatmap over the floor plan
        st.subheader("Coverage Heatmap")

        # Centered LaTeX Equation for Coverage Heatmap
        st.latex(r"\text{RSRP}(d, t) = 33 \times \left( -2.166 - \log_{10}(d) \right)")

        # Controls and Plot Side by Side
        col5, col6 = st.columns([3, 1], gap="small")
        with col6:
            st.write("**Heatmap Configuration**")
            # Additional controls can be added here if needed

        with col5:
            fig, ax = plt.subplots(figsize=(10, 7))
            # Show the floor plan
            ax.imshow(floor_plan_image, alpha=0.7)

            # We'll need an extent so that coverage_map aligns with the floor plan.
            # coverage_map has shape (map_height, map_width),
            # which corresponds to [0..map_width*downsample_factor] in x
            # and [0..map_height*downsample_factor] in y.
            extent = [0, map_width * downsample_factor, map_height * downsample_factor, 0]

            # Show coverage map
            cax = ax.imshow(
                coverage_map,
                cmap="jet",
                alpha=0.5,
                extent=extent,
                vmin=rsrp_min,
                vmax=rsrp_max
            )

            # Mark the gNB location (unscaled) as well
            ax.plot(gNB_x, gNB_y, 'wo', markersize=10, label='gNB')

            ax.set_title("Coverage Heatmap", fontsize=16)
            ax.invert_yaxis()  # Because extent's top=0, bottom=map_height*downsample_factor
            ax.legend(fontsize=12)

            # Add colorbar with adjusted padding
            fig.colorbar(cax, ax=ax, shrink=0.6, pad=0.02, label="RSRP (dB)")

            # Adjust tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=12)

            st.pyplot(fig)

        # 3.8 Identify “Red Zones”
        st.subheader("Identify Red Zones")

        # Centered LaTeX Equation for Red Zones Criteria
        st.latex(r"\text{Red Zone Criteria:} \quad \text{RSRP}(d) < \text{Threshold}")

        red_threshold = st.number_input("RSRP Threshold for Red Zones (dB)", value=-110.0, step=5.0, key="red_threshold_input")

        # Count how many grid cells are below threshold
        below_thresh = np.nansum(coverage_map < red_threshold)
        total_valid = np.count_nonzero(~np.isnan(coverage_map))
        st.markdown(f"**Grid cells below {red_threshold} dB:** {int(below_thresh)}")
        st.markdown(f"**Total valid coverage cells:** {int(total_valid)}")
        if total_valid > 0:
            ratio = below_thresh / total_valid * 100
            st.markdown(f"**Percentage of area in 'red zone':** {ratio:.2f} %")

        # Optional: Visualize Red Zones on the Heatmap
        with st.expander("Visualize Red Zones on Heatmap", expanded=False):
            # Centered LaTeX Equation for Red Zones Visualization
            st.latex(r"\text{Red Zones Highlighted}")

            # Create a mask for red zones
            red_zone_mask = coverage_map < red_threshold

            # Upsample the red_zone_mask to match the original image size for visualization
            red_zone_mask_upsampled = np.kron(red_zone_mask, np.ones((downsample_factor, downsample_factor)))

            # Ensure the mask matches the image dimensions
            red_zone_mask_upsampled = red_zone_mask_upsampled[:h, :w]

            # Convert mask to an image with semi-transparent red
            red_overlay = Image.new("RGBA", (w, h), (255, 0, 0, 0))
            draw = ImageDraw.Draw(red_overlay)
            red_pixels = np.argwhere(red_zone_mask_upsampled)
            for y, x in red_pixels:
                draw.point((x, y), fill=(255, 0, 0, 100))  # Semi-transparent red

            # Combine floor plan with red zones
            combined_image = floor_plan_image.copy()
            combined_image = Image.alpha_composite(combined_image, red_overlay)

            # Plot the combined image
            fig_red, ax_red = plt.subplots(figsize=(10, 7))
            ax_red.imshow(combined_image)
            ax_red.plot(gNB_x, gNB_y, 'wo', markersize=10, label='gNB')
            ax_red.set_title("Coverage Heatmap with Red Zones", fontsize=16)
            ax_red.legend(fontsize=12)
            ax_red.tick_params(axis='both', which='major', labelsize=12)

            st.pyplot(fig_red)
