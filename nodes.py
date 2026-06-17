# Tencent Hunyuan 3D 2.1 — Shape Generation Nodes
# Licensed under Tencent Hunyuan 3D 2.1 Community License Agreement

from PIL import Image, ImageSequence, ImageOps
import torch
import os
import numpy as np
import trimesh as Trimesh
import gc

from .hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from .hy3dshape.hy3dshape.postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover
from .hy3dshape.hy3dshape.rembg import BackgroundRemover
from .hy3dshape.hy3dshape.models.autoencoders import ShapeVAE
from typing import Optional
from pathlib import Path

import folder_paths
import node_helpers
import hashlib

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def parse_string_to_int_list(number_string):
    if not number_string:
        return []
    try:
        return [int(num.strip()) for num in number_string.split(",")]
    except ValueError as e:
        print(f"Error converting string to integer: {e}")
        return []

def get_picture_files(folder_path):
    picture_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []
    files = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            _, ext = os.path.splitext(entry)
            if ext.lower().endswith(picture_extensions):
                files.append(full_path)
    return files

def get_filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


# ══════════════════════════════════════════════
# Node: Mesh Generator (DiT)
# ══════════════════════════════════════════════

class Hy3DMeshGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "DiT model from ComfyUI/models/diffusion_models/"}),
                "image": ("IMAGE", {"tooltip": "Image to generate mesh from"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "tooltip": "Number of diffusion denoising steps. More steps = higher quality but slower. 50 is a good default"}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1, "max": 30, "step": 0.1, "tooltip": "How closely to follow the input image. Higher = more faithful but may reduce diversity. 5.0 recommended"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed for reproducible generation. Same seed + same inputs = same output"}),
                "attention_mode": (["sdpa", "sageattn"], {"default": "sdpa", "tooltip": "Attention implementation. sdpa is standard PyTorch, sageattn uses SageAttention for potential speedup"}),
            },
        }

    RETURN_TYPES = ("HY3DLATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DShape"
    DESCRIPTION = "Generate 3D mesh latents from an image using DiT flow-matching diffusion."

    def loadmodel(self, model, image, steps, guidance_scale, seed, attention_mode):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        seed = seed % (2**32)

        model_path = folder_paths.get_full_path("diffusion_models", model)

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            config_path=os.path.join(script_directory, "configs", "dit_config_2_1.yaml"),
            ckpt_path=model_path,
            device=device,
            offload_device=offload_device,
            attention_mode=attention_mode,
        )

        image = tensor2pil(image)

        latents = pipeline(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed),
        )

        del pipeline
        mm.soft_empty_cache()
        gc.collect()

        return (latents,)


# ══════════════════════════════════════════════
# Node: VAE Loader
# ══════════════════════════════════════════════

class Hy3D21VAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "VAE from ComfyUI/models/vae/"}),
            },
            "optional": {
                "vae_config": ("HY3D21VAECONFIG", {"tooltip": "Optional custom VAE configuration. Leave unconnected for default settings"}),
            },
        }

    RETURN_TYPES = ("HY3DVAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "loadvae"
    CATEGORY = "Hunyuan3DShape"
    DESCRIPTION = "Load a ShapeVAE model for decoding 3D mesh latents."

    def loadvae(self, model_name, vae_config=None):
        model_path = folder_paths.get_full_path("vae", model_name)

        if vae_config is not None:
            vae = ShapeVAE(**vae_config)
        else:
            vae = ShapeVAE(
                num_latents=4096,
                embed_dim=64,
                num_freqs=8,
                include_pi=False,
                heads=16,
                width=1024,
                num_encoder_layers=8,
                num_decoder_layers=16,
                qkv_bias=False,
                qk_norm=True,
                scale_factor=1.0039506158752403,
                geo_decoder_mlp_expand_ratio=4,
                geo_decoder_downsample_ratio=1,
                geo_decoder_ln_post=True,
                point_feats=4,
                pc_size=81920,
                pc_sharpedge_size=0,
            )

        sd = load_torch_file(model_path)
        vae.load_state_dict(sd, strict=False)
        return (vae,)


# ══════════════════════════════════════════════
# Node: VAE Config
# ══════════════════════════════════════════════

class Hy3D21VAEConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_latents": ("INT", {"default": 4096, "tooltip": "Number of latent tokens. Default 4096 matches the pretrained model"}),
                "embed_dim": ("INT", {"default": 64, "tooltip": "Embedding dimension per latent token"}),
                "num_freqs": ("INT", {"default": 8, "tooltip": "Positional encoding frequencies for 3D coordinate inputs"}),
                "include_pi": ("BOOLEAN", {"default": False, "tooltip": "Include pi in positional encoding frequency bands"}),
                "heads": ("INT", {"default": 16, "tooltip": "Number of attention heads in transformer layers"}),
                "width": ("INT", {"default": 1024, "tooltip": "Hidden dimension width of transformer layers"}),
                "num_encoder_layers": ("INT", {"default": 8, "tooltip": "Number of transformer layers in the encoder"}),
                "num_decoder_layers": ("INT", {"default": 16, "tooltip": "Number of transformer layers in the decoder"}),
                "qkv_bias": ("BOOLEAN", {"default": False, "tooltip": "Add bias terms to query/key/value projections"}),
                "qk_norm": ("BOOLEAN", {"default": True, "tooltip": "Apply RMS normalization to query and key vectors"}),
                "scale_factor": ("FLOAT", {"default": 1.0039506158752403, "tooltip": "Latent space scaling factor. Must match the pretrained checkpoint"}),
                "geo_decoder_mlp_expand_ratio": ("INT", {"default": 4, "tooltip": "MLP expansion ratio in geometry decoder"}),
                "geo_decoder_downsample_ratio": ("INT", {"default": 1, "tooltip": "Spatial downsampling ratio in geometry decoder"}),
                "geo_decoder_ln_post": ("BOOLEAN", {"default": True, "tooltip": "Apply layer normalization after geometry decoder"}),
                "point_feats": ("INT", {"default": 4, "tooltip": "Number of feature channels per 3D point"}),
                "pc_size": ("INT", {"default": 81920, "tooltip": "Total number of points in the decoded point cloud (81920 = 320x256)"}),
                "pc_sharpedge_size": ("INT", {"default": 0, "tooltip": "Additional points for sharp edge preservation. 0 = disabled"}),
            },
        }

    RETURN_TYPES = ("HY3D21VAECONFIG",)
    RETURN_NAMES = ("vae_config",)
    FUNCTION = "create_config"
    CATEGORY = "Hunyuan3DShape"
    DESCRIPTION = "Override ShapeVAE configuration parameters."

    def create_config(self, **kwargs):
        return (kwargs,)


# ══════════════════════════════════════════════
# Node: VAE Decode
# ══════════════════════════════════════════════

class Hy3D21VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("HY3DVAE", {"tooltip": "ShapeVAE model from the VAE Loader node"}),
                "latents": ("HY3DLATENT", {"tooltip": "3D latent codes from the Mesh Generator node"}),
                "box_v": ("FLOAT", {"default": 1.01, "min": 0.5, "max": 5.0, "step": 0.01, "tooltip": "Bounding box scale for marching cubes. Larger values capture more of the shape but reduce resolution"}),
                "octree_resolution": ("INT", {"default": 256, "min": 16, "max": 512, "step": 16, "tooltip": "Voxel grid resolution for surface extraction. Higher = finer detail but more memory and time"}),
                "mc_level": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Isosurface level for marching cubes. 0.0 is standard. Adjust to grow (+) or shrink (-) the surface"}),
                "mc_algo": (["mc", "dmc"], {"default": "mc", "tooltip": "Surface extraction algorithm. mc = Marching Cubes (standard), dmc = Dual Marching Cubes (preserves sharp features)"}),
                "num_chunks": ("INT", {"default": 8000, "min": 1, "max": 500000, "step": 1000, "tooltip": "Split decoding into chunks to reduce memory usage. Lower = less memory but slower"}),
            },
            "optional": {
                "flash_vdm": ("BOOLEAN", {"default": False, "tooltip": "Enable flash volume decoding for faster inference (experimental)"}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "decode"
    CATEGORY = "Hunyuan3DShape"
    DESCRIPTION = "Decode diffusion latents to a 3D mesh via marching cubes."

    def decode(self, vae, latents, box_v, octree_resolution, mc_level, mc_algo, num_chunks, flash_vdm=False):
        device = mm.get_torch_device()

        vae.eval()
        vae.to(device)

        # Cast latents to match VAE's dtype (fp16 weights + fp32 latents fails on MPS)
        vae_dtype = next(vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)

        # Pass through VAE forward: post_kl maps embed_dim(64) → width(1024), then transformer
        latents = vae(latents)

        mesh_output = vae.latents2mesh(
            latents,
            bounds=box_v,
            octree_resolution=octree_resolution,
            mc_level=mc_level,
            mc_algo=mc_algo,
            num_chunks=num_chunks,
        )

        # latents2mesh returns a list of Latent2MeshOutput (one per batch item)
        mesh_output = mesh_output[0]
        print(f"Decoded mesh with {mesh_output.mesh_v.shape[0]} vertices and {mesh_output.mesh_f.shape[0]} faces")

        mesh = Trimesh.Trimesh(vertices=mesh_output.mesh_v, faces=mesh_output.mesh_f)
        # Marching cubes can produce inward-facing normals — flip them outward
        if mesh.volume < 0:
            mesh.invert()
        vae.to("cpu")
        mm.soft_empty_cache()
        gc.collect()

        return (mesh,)


# ══════════════════════════════════════════════
# Node: Resize Images
# ══════════════════════════════════════════════

class Hy3D21ResizeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to resize"}),
                "width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "tooltip": "Target width in pixels"}),
                "height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "tooltip": "Target height in pixels"}),
                "sampling": (["nearest", "bilinear", "bicubic", "lanczos"], {"tooltip": "Interpolation method. lanczos is highest quality, nearest is fastest"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize"
    CATEGORY = "Hunyuan3DShape"
    DESCRIPTION = "Resize images with configurable interpolation method."

    def resize(self, image, width, height, sampling):
        interp_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        pil = tensor2pil(image)
        pil = pil.resize((width, height), interp_map[sampling])
        return (pil2tensor(pil),)


# ══════════════════════════════════════════════
# Node: Load Image with Transparency
# ══════════════════════════════════════════════

class Hy3D21LoadImageWithTransparency:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True, "tooltip": "Image file to load from ComfyUI input directory"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "image_with_alpha")
    FUNCTION = "load_image"
    CATEGORY = "Hunyuan3DShape"
    DESCRIPTION = "Load an image and extract its alpha channel as a mask."

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)
        output_images = []
        output_masks = []
        output_images_ori = []
        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            output_images_ori.append(pil2tensor(i))

            if i.mode == "I":
                i = i.point(lambda x: x * (1 / 255))
            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask = np.array(i.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
            output_image_ori = torch.cat(output_images_ori, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            output_image_ori = output_images_ori[0]

        return (output_image, output_mask, output_image_ori)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


# ══════════════════════════════════════════════
# Node: Mesh Generation Batch
# ══════════════════════════════════════════════

class Hy3D21MeshGenerationBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_folder": ("STRING", {"tooltip": "Folder containing input images (jpg, png, etc.)"}),
                "output_folder": ("STRING", {"tooltip": "Folder where generated meshes will be saved as GLB files"}),
                "vae_model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "ShapeVAE checkpoint for decoding latents to mesh"}),
                "dit_model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "DiT diffusion model for generating 3D latents from images"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "tooltip": "Number of diffusion steps per image. More = better quality, slower"}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1, "max": 30, "step": 0.1, "tooltip": "How closely to follow each input image. 5.0 recommended"}),
                "box_v": ("FLOAT", {"default": 1.01, "min": 0.5, "max": 5.0, "step": 0.01, "tooltip": "Bounding box scale for marching cubes extraction"}),
                "octree_resolution": ("INT", {"default": 256, "min": 16, "max": 512, "step": 16, "tooltip": "Voxel resolution for surface extraction. Higher = finer detail"}),
                "mc_level": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Isosurface level. 0.0 is standard"}),
                "mc_algo": (["mc", "dmc"], {"default": "mc", "tooltip": "mc = Marching Cubes, dmc = Dual Marching Cubes"}),
                "num_chunks": ("INT", {"default": 8000, "min": 1, "max": 500000, "step": 1000, "tooltip": "Decode in chunks to reduce memory. Lower = less memory"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "tooltip": "Base random seed. Overridden if generate_random_seed is enabled"}),
                "generate_random_seed": ("BOOLEAN", {"default": True, "tooltip": "Use a random seed for each image instead of the fixed seed"}),
                "simplify": ("BOOLEAN", {"default": True, "tooltip": "Reduce face count after generation using quadric decimation"}),
                "max_facenum": ("INT", {"default": 40000, "min": 1, "max": 10000000, "step": 1, "tooltip": "Maximum faces when simplify is enabled"}),
                "remove_background": ("BOOLEAN", {"default": False, "tooltip": "Remove image background before generation (requires rembg)"}),
                "attention_mode": (["sdpa", "sageattn"], {"default": "sdpa", "tooltip": "sdpa = standard, sageattn = SageAttention"}),
                "skip_generated_mesh": ("BOOLEAN", {"default": True, "tooltip": "Skip images that already have a generated GLB in the output folder"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("input_folder", "output_folder", "processed_images", "processed_meshes")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DShape"
    DESCRIPTION = "Batch process a folder of images to generate 3D meshes."
    OUTPUT_NODE = True

    def process(self, input_folder, output_folder, vae_model_name, dit_model_name, steps, guidance_scale,
                box_v, octree_resolution, mc_level, mc_algo, num_chunks, seed, generate_random_seed,
                simplify, max_facenum, remove_background, attention_mode, skip_generated_mesh):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        rembg = BackgroundRemover()

        processed_input_images = []
        processed_output_meshes = []

        files = get_picture_files(input_folder)
        nb_pictures = len(files)

        if nb_pictures > 0:
            dit_model_path = folder_paths.get_full_path("diffusion_models", dit_model_name)

            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                config_path=os.path.join(script_directory, "configs", "dit_config_2_1.yaml"),
                ckpt_path=dit_model_path,
                device=device,
                offload_device=offload_device,
                attention_mode=attention_mode,
            )

            vae_model_path = folder_paths.get_full_path("vae", vae_model_name)
            vae = ShapeVAE(
                num_latents=4096, embed_dim=64, num_freqs=8, include_pi=False,
                heads=16, width=1024, num_encoder_layers=8, num_decoder_layers=16,
                qkv_bias=False, qk_norm=True, scale_factor=1.0039506158752403,
                geo_decoder_mlp_expand_ratio=4, geo_decoder_downsample_ratio=1,
                geo_decoder_ln_post=True, point_feats=4, pc_size=81920, pc_sharpedge_size=0,
            )
            sd = load_torch_file(vae_model_path)
            vae.load_state_dict(sd, strict=False)
            vae.eval()
            vae.to(device)

            pbar = ProgressBar(nb_pictures)

            for file in files:
                file_name = get_filename_without_extension(file)
                output_glb_path = os.path.join(output_folder, f"{file_name}.glb")

                if skip_generated_mesh and os.path.exists(output_glb_path):
                    print(f"Skipping file {file}")
                    pbar.update(1)
                    continue

                print(f"Processing {file} ...")

                image = Image.open(file)
                if remove_background:
                    print("Removing background ...")
                    image = rembg(image)

                if generate_random_seed:
                    seed = int.from_bytes(os.urandom(4), "big")

                latents = pipeline(
                    image=image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=torch.manual_seed(seed),
                )

                # Cast latents to match VAE's dtype (fp16 weights + fp32 latents fails on MPS)
                vae_dtype = next(vae.parameters()).dtype
                latents = latents.to(dtype=vae_dtype)

                # Pass through VAE forward: post_kl maps embed_dim(64) → width(1024), then transformer
                latents = vae(latents)

                mesh_output = vae.latents2mesh(
                    latents, bounds=box_v, octree_resolution=octree_resolution,
                    mc_level=mc_level, mc_algo=mc_algo, num_chunks=num_chunks,
                )

                mesh = Trimesh.Trimesh(vertices=mesh_output[0].mesh_v, faces=mesh_output[0].mesh_f)
                # Marching cubes can produce inward-facing normals — flip them outward
                if mesh.volume < 0:
                    mesh.invert()

                if simplify:
                    current_faces_num = mesh.faces.shape[0]
                    print(f"Current Faces Number: {current_faces_num}")
                    if current_faces_num > max_facenum:
                        print("Decimating ...")
                        mesh = FaceReducer()(mesh, max_facenum=max_facenum)

                os.makedirs(output_folder, exist_ok=True)
                mesh.export(output_glb_path, file_type="glb")

                processed_input_images.append(file)
                processed_output_meshes.append(output_glb_path)
                pbar.update(1)

                mm.soft_empty_cache()
                gc.collect()

            del pipeline
            del vae
            mm.soft_empty_cache()
            gc.collect()

        return (input_folder, output_folder, processed_input_images, processed_output_meshes)


# ══════════════════════════════════════════════
# Registration
# ══════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "Hy3DMeshGenerator": Hy3DMeshGenerator,
    "Hy3D21VAELoader": Hy3D21VAELoader,
    "Hy3D21VAEConfig": Hy3D21VAEConfig,
    "Hy3D21VAEDecode": Hy3D21VAEDecode,
    "Hy3D21ResizeImages": Hy3D21ResizeImages,
    "Hy3D21LoadImageWithTransparency": Hy3D21LoadImageWithTransparency,
    "Hy3D21MeshGenerationBatch": Hy3D21MeshGenerationBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DMeshGenerator": "Hunyuan 3D 2.1 Mesh Generator",
    "Hy3D21VAELoader": "Hunyuan 3D 2.1 VAE Loader",
    "Hy3D21VAEConfig": "Hunyuan 3D 2.1 VAE Config",
    "Hy3D21VAEDecode": "Hunyuan 3D 2.1 VAE Decoder",
    "Hy3D21ResizeImages": "Hunyuan 3D 2.1 Resize Images",
    "Hy3D21LoadImageWithTransparency": "Hunyuan 3D 2.1 Load Image with Transparency",
    "Hy3D21MeshGenerationBatch": "Hunyuan 3D 2.1 Mesh Generator from Folder",
}
