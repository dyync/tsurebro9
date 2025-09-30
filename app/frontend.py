import gradio as gr
from PIL import Image
from app.pipeline import preprocess_image, preprocess_images, image_to_3d, get_seed

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_prompt = gr.Image(type="pil", label="Single Image")
            multiimage_prompt = gr.Gallery(type="pil", label="Multi Images")
            seed = gr.Slider(0, 2**31-1, value=0)
            randomize_seed = gr.Checkbox(value=True)
            generate_btn = gr.Button("Generate")

        with gr.Column():
            video_output = gr.Video()
            download_btn = gr.DownloadButton(label="Download Video")

    generate_btn.click(
        fn=lambda img, imgs, rand, s: image_to_3d(
            image=img,
            multiimages=imgs,
            is_multiimage=len(imgs) > 0,
            seed=get_seed(rand, s),
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3.0,
            slat_sampling_steps=12,
            multiimage_algo="stochastic",
            session_hash="demo"
        ),
        inputs=[image_prompt, multiimage_prompt, randomize_seed, seed],
        outputs=[video_output]
    )

if __name__ == "__main__":
    demo.launch()
