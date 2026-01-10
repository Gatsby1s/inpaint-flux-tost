import os, json, requests, runpod
import random, time, base64
import torch
import numpy as np
from PIL import Image
import nodes
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_custom_sampler, nodes_flux, nodes_controlnet
from comfy import model_management

DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
ControlNetLoader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()

# Fix for nodes_flux compatibility
try:
    FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
except (AttributeError, KeyError):
    FluxGuidance = nodes_flux.FluxGuidance()

# Fix for nodes_custom_sampler compatibility
try:
    RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
    BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
    KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
    BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
    SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
except (AttributeError, KeyError):
    RandomNoise = nodes_custom_sampler.RandomNoise()
    BasicGuider = nodes_custom_sampler.BasicGuider()
    KSamplerSelect = nodes_custom_sampler.KSamplerSelect()
    BasicScheduler = nodes_custom_sampler.BasicScheduler()
    SamplerCustomAdvanced = nodes_custom_sampler.SamplerCustomAdvanced()

VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
ImageScaleBy = NODE_CLASS_MAPPINGS["ImageScaleBy"]()

# Fix for nodes_controlnet compatibility
try:
    ControlNetInpaintingAliMamaApply = nodes_controlnet.NODE_CLASS_MAPPINGS["ControlNetInpaintingAliMamaApply"]()
except (AttributeError, KeyError):
    ControlNetInpaintingAliMamaApply = nodes_controlnet.ControlNetInpaintingAliMamaApply()

with torch.inference_mode():
    clip = DualCLIPLoader.load_clip("t5xxl_fp16.safetensors", "clip_l.safetensors", "flux")[0]
    unet = UNETLoader.load_unet("flux1-dev.sft", "default")[0]
    vae = VAELoader.load_vae("ae.sft")[0]
    controlnet = ControlNetLoader.load_controlnet("FLUX.1-dev-Controlnet-Inpainting-Beta.safetensors")[0]

def download_file(url, save_dir='/content/ComfyUI/input'):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@torch.inference_mode()
def generate(input):
    values = input["input"]

    # Support both base64 and URL input
    if 'input_mask_base64' in values:
        # Decode base64 image
        image_data = base64.b64decode(values['input_mask_base64'])
        input_image = '/content/ComfyUI/input/temp_input.png'
        os.makedirs('/content/ComfyUI/input', exist_ok=True)
        with open(input_image, 'wb') as f:
            f.write(image_data)
    else:
        # Fallback to URL download
        input_mask = values.get('input_mask', '')
        input_image = download_file(input_mask)
    
    width, height = Image.open(input_image).size
    
    positive_prompt = values.get('positive_prompt', 'clean background, no watermark, seamless')
    negative_prompt = values.get('negative_prompt', 'watermark, text, logo')
    controlnet_strength = values.get('controlnet_strength', 0.95)
    seed = values.get('seed', 0)
    steps = values.get('steps', 20)
    guidance = values.get('guidance', 30)
    sampler_name = values.get('sampler_name', 'euler')
    scheduler = values.get('scheduler', 'simple')
    cfg = values.get('cfg', 1.0)

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(f"Using seed: {seed}")

    cond = nodes.CLIPTextEncode().encode(clip, positive_prompt)[0]
    cond = FluxGuidance.append(cond, guidance)[0]
    n_cond = nodes.CLIPTextEncode().encode(clip, negative_prompt)[0]
    input_image_tensor, input_mask = LoadImage.load_image(input_image)
    latent_image = EmptyLatentImage.generate(closestNumber(width, 8), closestNumber(height, 8))[0]
    positive, negative = ControlNetInpaintingAliMamaApply.apply_inpaint_controlnet(
        positive=cond, negative=n_cond, control_net=controlnet, 
        image=input_image_tensor, mask=input_mask, strength=controlnet_strength, 
        vae=vae, start_percent=0, end_percent=1
    )
    sample = nodes.common_ksampler(
        model=unet, seed=seed, steps=steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative,
        latent=latent_image, denoise=1.0
    )[0]
    decoded = VAEDecode.decode(vae, sample)[0].detach()
    
    output_path = "/content/inpaint-flux-output.png"
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(output_path)

    # Return base64 encoded image
    with open(output_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)

    return {
        "image_base64": image_base64,
        "seed": seed,
        "width": width,
        "height": height
    }

runpod.serverless.start({"handler": generate})
