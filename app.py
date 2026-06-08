import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stabilityai/sd-turbo"

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

STYLE_PREFIX = """
clearly hand-drawn, unique weird cute creature, folk art monster,
painted illustration, visible brush texture, not 3d, not glossy, not bubble animation,
quirky, charming, strange, expressive
"""

NEGATIVE_PROMPT = """
3d, glossy, plastic, bubble style, realistic human, gore, violence, horror, blurry,
extra text, watermark, logo
"""

def generate_creature(user_prompt):
    full_prompt = f"{STYLE_PREFIX}. {user_prompt}"

    image = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=512,
        width=512
    ).images[0]

    return image

with gr.Blocks(title="Whispervale Creature Generator") as demo:
    gr.Markdown("# Whispervale Creature Generator")
    gr.Markdown("Create weird, cute, hand-drawn monster characters.")

    prompt = gr.Textbox(
        label="Describe your creature",
        value="frog alien character, square green head body, enlarged irregular nostrils, huge white eyes with tiny black pupils, many tiny legs, wild grin with imperfect white teeth",
        lines=4
    )

    button = gr.Button("Generate Creature")
    output = gr.Image(label="Generated Creature", type="pil")

    button.click(
        fn=generate_creature,
        inputs=prompt,
        outputs=output
    )

demo.launch()
