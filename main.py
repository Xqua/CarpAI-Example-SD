from diffusers import DiffusionPipeline
import torch
from PIL import Image
import os, sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Stable Diffusion")
parser.add_argument("--h", dest="height", type=int,help="height of the image",default=512)
parser.add_argument("--w", dest="width", type=int,help="width of the image",default=512)
parser.add_argument("--p", dest="prompt", type=str,help="Description of the image you want to generate",default="cat")
parser.add_argument("--n", dest="numSteps", type=int,help="Number of Steps",default=50)
parser.add_argument("--s", dest="seed", type=int,help="Seed",default=-1)
parser.add_argument("--g", dest="guidanceScale", type=float,help="Number of Steps",default=7.5)
parser.add_argument("--b", dest="batchSize", type=int,help="Number of Images",default=1)
parser.add_argument("--o", dest="output", type=str,help="Output Folder where to store the Image",default="./")

args=parser.parse_args()
height=args.height
width=args.width
prompt=args.prompt
numSteps=args.numSteps
seed=args.seed
guidanceScale=args.guidanceScale
batchSize=args.batchSize
output=args.output

if seed == -1:
    seed = np.random.randint(0, 10000000)

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(seed)

images = pipeline(prompt,
                  guidance_scale=guidanceScale,
                  num_inference_steps=numSteps,
                  height=height, 
                  width=width,
                  generator=generator
                  )
print(images)

for img in images:
    pil_img = Image.fromarray(img)
    pil_img.save(f"{output}/image_{i}.png")
