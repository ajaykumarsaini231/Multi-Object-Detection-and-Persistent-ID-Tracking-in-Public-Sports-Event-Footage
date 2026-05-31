import gradio as gr
import subprocess
import tempfile
import os

def track_video(video):

    output_video = "outputs/tracked.mp4"

    cmd = [
        "python",
        "run_pipeline.py",
        "--input",
        video,
        "--output",
        output_video,
        "--heatmap",
        "--count-chart",
        "--speed"
    ]

    subprocess.run(cmd, check=True)

    return (
        output_video,
        "outputs/heatmap.png",
        "outputs/object_count.png"
    )

demo = gr.Interface(
    fn=track_video,
    inputs=gr.Video(label="Upload Sports/Event Video"),
    outputs=[
        gr.Video(label="Tracked Video"),
        gr.Image(label="Heatmap"),
        gr.Image(label="Object Count")
    ],
    title="Multi-Object Detection & Persistent ID Tracking",
    description="YOLOv8 + ByteTrack"
)

demo.launch()