from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda:0")

prompts = ["A chair in 2d style",
           "A chair in flat style"]

# content = "A painting of baby penguin"
# style = "in the style of melting golden 3D rendering"

# prompt_1 = "Melting golden 3D rendering style"
# prompt_2 = "a baby penguin"

# image = pipe(prompt=prompt_1, prompt_2=prompt_2, num_inference_steps=30, guidance_scale=7.5).images[0]
# image.save("style_1.png")
for id, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, prompt_2=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"outcomes/sdxl/{prompt}.png")
