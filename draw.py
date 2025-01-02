import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

# Load the floor plan
def load_floor_plan(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img

# Draw rectangles and circles interactively
def draw_zones(image):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title('Draw Zones | LMB: Green (Rect), RMB: Red (Rect), MMB: Blue (Circle)')

    zones = []
    rect = None
    start_point = None

    # Mouse press event
    def on_press(event):
        nonlocal rect, start_point
        if event.inaxes != ax:  # Ignore clicks outside the image
            return

        start_point = (event.xdata, event.ydata)

        # Draw rectangles for LMB/RMB and circles for MMB
        if event.button == 1:
            color = 'green'
            rect = Rectangle(start_point, 1, 1, linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
            zones.append((rect, color, 'rectangle'))

        elif event.button == 3:
            color = 'red'
            rect = Rectangle(start_point, 1, 1, linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
            zones.append((rect, color, 'rectangle'))

        elif event.button == 2:  # Middle click for blue circle
            radius = 15  # Fixed circle radius
            circle = Circle(start_point, radius, edgecolor='blue', facecolor='blue', alpha=0.5)
            ax.add_patch(circle)
            zones.append((circle, 'blue', 'circle'))

        plt.draw()

    # Mouse drag event (resize rectangle)
    def on_motion(event):
        nonlocal rect
        if rect is None or start_point is None:
            return
        x0, y0 = start_point
        width = event.xdata - x0
        height = event.ydata - y0
        rect.set_width(width)
        rect.set_height(height)
        plt.draw()

    # Mouse release event (finalize rectangle)
    def on_release(event):
        nonlocal rect, start_point
        rect = None
        start_point = None
        plt.draw()

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.show()
    return zones

# Main execution
image_path = 'maps/refinery_floorplan.jpg'
floor_plan = load_floor_plan(image_path)
zones = draw_zones(floor_plan)

# Output zone details
for zone, color, shape in zones:
    if shape == 'rectangle':
        print(f"Rectangle - x={zone.get_x()}, y={zone.get_y()}, w={zone.get_width()}, h={zone.get_height()}, color={color}")
    elif shape == 'circle':
        print(f"Circle - x={zone.center[0]}, y={zone.center[1]}, radius={zone.radius}, color={color}")
