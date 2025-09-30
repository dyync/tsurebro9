def render_video(obj, num_frames=120):
    # Stub: Replace with actual rendering logic
    return {
        "color": [obj.render_frame(i) for i in range(num_frames)],
        "normal": [obj.render_normal(i) for i in range(num_frames)]
    }
