import configparser
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

import yaml
import os
import sys
from tqdm import tqdm
import logging

os.getcwd()
config = configparser.ConfigParser()
config.read("config.ini")
access_token = config.get("huggingface","huggingface_access_token")
logger = logging.getLogger()
Log_Format = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=Log_Format)
# hf_hub_download(repo_id="CompVis/stable-diffusion-v1-4", use_auth_token=access_token)
# model_id = "CompVis/ldm-text2im-large-256"
model_id = "CompVis/stable-diffusion-v1-4"
logger.info(f"Looking for model {model_id}")

# load model and scheduler
# pipe = DiffusionPipeline.from_pretrained(model_id, use_auth_token=access_token)
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, use_auth_token=access_token
)
logger.info(f"Downloaded {model_id}")

# run pipeline in inference (sample random noise and denoise)
num_images = 1
prompt = [
    "A blond female warrior with an axe and war paint on her face, closeup realistic photograph"
] * num_images
folder_path = "/Users/tjpajala/Cambri/cambri2/analytics/notebooks/stable_diffusion/output"
logger.info(f"Creating {num_images} images")
# images = pipe(prompt, num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]
images = pipe(prompt, num_inference_steps=20, guidance_scale=7.5)["sample"]
# save images
logger.info(f"Saving {num_images} images")
for idx, image in tqdm(enumerate(images), total=num_images):
    filename = prompt[idx].replace(",", "").replace(" ", "-")
    image.save(f"{folder_path}/{filename}-{idx}.png")

