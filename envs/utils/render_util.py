import glfw
# Define your custom setup function for the viewer
def setup_viewer(viewer):
    # Set real-time rendering
    viewer._run_speed = 1.0  # Run at real time
    viewer._render_every_frame = False  # Don't render every frame
    viewer._hide_menu = True  # Hide the bar
    # import ipdb; ipdb.set_trace()
    
    # Set the window size
    if hasattr(viewer, "window"):
        window = viewer.window
        glfw.set_window_size(window, 800, 600)
    
    # Hide the bar