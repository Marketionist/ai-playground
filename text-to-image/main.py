import os
from diffusers import StableDiffusionPipeline
import torch

model_id = 'sd-legacy/stable-diffusion-v1-5'
has_cuda = torch.cuda.is_available()
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16 if has_cuda else torch.float32
)
# Use GPU or processor depending on device
device = torch.device('cuda' if has_cuda else 'cpu')
print(f'Using device: {device}')
pipe = pipe.to(device)

prompt = os.getenv('PROMPT')
image = pipe(prompt).images[0]

image.save('image_result.png')
