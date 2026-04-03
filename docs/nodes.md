# ComfyUI-Hunyuan3D-Shape Node Reference

## Mesh Generator

The main shape generation node. Takes a single image and produces 3D mesh latents using a Diffusion Transformer (DiT) with flow-matching. The latents encode the 3D shape and must be decoded by the VAE Decode node to produce an actual mesh.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | ENUM | - | DiT checkpoint from `ComfyUI/models/diffusion_models/`. The standard model is `hunyuan3d-dit-v2-1.ckpt` (~7.4 GB). An fp16 variant is also available for lower memory usage. |
| `image` | IMAGE | - | Reference image of the object to generate in 3D. Works best with a single object on a clean background. RGBA images with transparency are handled automatically. |
| `steps` | INT | `50` | Number of diffusion denoising steps. The model progressively refines the 3D latent from noise. **Lower (20-30):** faster but rougher shapes. **Default (50):** good quality/speed balance. **Higher (75-100):** diminishing returns, mainly useful for complex shapes. |
| `guidance_scale` | FLOAT | `5.0` | Classifier-free guidance scale. Controls how strongly the generation follows the input image. **Low (1-3):** more creative/diverse but may drift from the image. **Default (5.0):** faithful to the image. **High (10+):** very strict adherence but can produce artifacts. |
| `seed` | INT | `0` | Random seed for reproducible results. The same seed with the same inputs will always produce the same output. Change the seed to get variations. |
| `attention_mode` | ENUM | `"sdpa"` | Attention implementation. **sdpa:** PyTorch's scaled dot-product attention (standard, reliable). **sageattn:** SageAttention library (may be faster on some hardware, requires separate installation). |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `latents` | HY3DLATENT | 3D shape latent codes (4096 tokens x 64 dims). Feed into the VAE Decode node. |

---

## VAE Loader

Load the ShapeVAE model that decodes 3D latents into point clouds and meshes.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | ENUM | - | VAE checkpoint from `ComfyUI/models/vae/`. The standard model is `hunyuan3d-vae-v2-1.ckpt` (~656 MB). |
| `vae_config` | HY3D21VAECONFIG | *(optional)* | Custom VAE configuration from the VAE Config node. Leave unconnected to use the default configuration that matches the pretrained checkpoint. Only needed for research or custom-trained models. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `vae` | HY3DVAE | The loaded VAE model, ready for the VAE Decode node. |

---

## VAE Config

Override the default ShapeVAE architecture parameters. This is an advanced node — most users should leave the VAE Loader unconnected and use defaults.

Only use this if you have a custom-trained VAE with different architecture settings.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_latents` | INT | `4096` | Number of latent tokens. Must match the DiT model's output. |
| `embed_dim` | INT | `64` | Dimension of each latent token embedding. |
| `num_freqs` | INT | `8` | Number of frequency bands for positional encoding of 3D coordinates. |
| `include_pi` | BOOLEAN | `False` | Include pi-scaled frequencies in positional encoding. |
| `heads` | INT | `16` | Number of attention heads in the transformer. |
| `width` | INT | `1024` | Hidden dimension of the transformer layers. |
| `num_encoder_layers` | INT | `8` | Transformer layers in the encoder (latent → features). |
| `num_decoder_layers` | INT | `16` | Transformer layers in the decoder (features → 3D). |
| `qkv_bias` | BOOLEAN | `False` | Add bias to query/key/value attention projections. |
| `qk_norm` | BOOLEAN | `True` | Apply RMS normalization to query/key vectors for training stability. |
| `scale_factor` | FLOAT | `1.004` | Latent space scaling factor. Must exactly match the pretrained checkpoint. |
| `geo_decoder_mlp_expand_ratio` | INT | `4` | MLP width multiplier in the geometry decoder. |
| `geo_decoder_downsample_ratio` | INT | `1` | Spatial downsampling in geometry decoder. 1 = no downsampling. |
| `geo_decoder_ln_post` | BOOLEAN | `True` | Apply layer normalization after the geometry decoder. |
| `point_feats` | INT | `4` | Feature channels per decoded 3D point. |
| `pc_size` | INT | `81920` | Total point cloud size (81920 = 320 x 256 grid). |
| `pc_sharpedge_size` | INT | `0` | Additional points allocated for sharp edge preservation. 0 = disabled. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `vae_config` | HY3D21VAECONFIG | Configuration dict to connect to the VAE Loader. |

---

## VAE Decode

Decode 3D latent codes into an actual triangle mesh using marching cubes surface extraction.

The process: latents are decoded into a volumetric occupancy field on an octree grid, then marching cubes extracts the isosurface as a triangle mesh.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vae` | HY3DVAE | - | ShapeVAE model from the VAE Loader node. |
| `latents` | HY3DLATENT | - | 3D latent codes from the Mesh Generator node. |
| `box_v` | FLOAT | `1.01` | Bounding box half-size for the extraction volume. **1.01** (default) tightly frames the shape. Increase if the shape is getting clipped at the edges. Decrease if the mesh has excessive empty space. |
| `octree_resolution` | INT | `256` | Voxel grid resolution for surface extraction. **128:** fast but blocky. **256:** good default, captures most detail. **384-512:** high detail but significantly more memory and time. Each doubling roughly 8x the compute. |
| `mc_level` | FLOAT | `0.0` | Isosurface threshold for marching cubes. **0.0** is the natural surface. **Positive values** expand/thicken the mesh. **Negative values** shrink/erode it. Useful for adjusting wall thickness for 3D printing. |
| `mc_algo` | ENUM | `"mc"` | Surface extraction algorithm. **mc** (Marching Cubes): standard algorithm, smooth surfaces. **dmc** (Dual Marching Cubes): better at preserving sharp edges and corners, slightly slower. |
| `num_chunks` | INT | `8000` | Split the decoding into this many chunks to reduce peak memory. **Lower** = less memory but slower. **Higher** = faster but more memory. Reduce if you get out-of-memory errors. |
| `flash_vdm` | BOOLEAN | `False` | Experimental flash volume decoding mode. May speed up inference on supported hardware. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `trimesh` | TRIMESH | The decoded 3D triangle mesh. Feed into MeshTools for post-processing, or directly into the Paint package for texturing. |

---

## Resize Images

Simple image resize utility with selectable interpolation.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | - | Input image to resize. |
| `width` | INT | `512` | Target width in pixels. |
| `height` | INT | `512` | Target height in pixels. |
| `sampling` | ENUM | - | Interpolation method. **nearest:** fastest, pixelated (good for masks). **bilinear:** smooth, fast. **bicubic:** smoother, slightly slower. **lanczos:** highest quality, best for downscaling. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The resized image. |

---

## Load Image with Transparency

Load an image file and separate it into RGB and alpha channels. Useful for preparing reference images with transparent backgrounds for the Mesh Generator.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | ENUM | - | Image file from ComfyUI's input directory. Supports PNG (with alpha), JPG, GIF, BMP, TIFF, WebP. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | RGB image with background (white where transparent). |
| `mask` | MASK | Alpha channel as a mask (1.0 = transparent, 0.0 = opaque). |
| `image_with_alpha` | IMAGE | Original image including alpha channel. |

---

## Mesh Generation Batch

Process an entire folder of images to generate 3D meshes automatically. Each image produces one GLB file. Supports skipping already-generated meshes for resumable batch runs.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_folder` | STRING | - | Folder containing input images (JPG, PNG, etc.). Each image generates one mesh. |
| `output_folder` | STRING | - | Where to save generated GLB files. Named after the input image (e.g., `chair.png` → `chair.glb`). |
| `vae_model_name` | ENUM | - | ShapeVAE checkpoint for decoding latents to mesh. |
| `dit_model_name` | ENUM | - | DiT model for image-to-3D generation. |
| `steps` | INT | `50` | Diffusion steps per image. More = better quality, slower. |
| `guidance_scale` | FLOAT | `5.0` | Image guidance strength. 5.0 recommended. |
| `box_v` | FLOAT | `1.01` | Bounding box scale for marching cubes. |
| `octree_resolution` | INT | `256` | Voxel resolution for surface extraction. |
| `mc_level` | FLOAT | `0.0` | Isosurface level (0.0 = standard). |
| `mc_algo` | ENUM | `"mc"` | mc = Marching Cubes, dmc = Dual Marching Cubes. |
| `num_chunks` | INT | `8000` | Memory chunking for VAE decode. |
| `seed` | INT | `0` | Base random seed. Overridden if `generate_random_seed` is on. |
| `generate_random_seed` | BOOLEAN | `True` | Generate a unique random seed for each image. Produces more variety. Disable for reproducible batches. |
| `simplify` | BOOLEAN | `True` | Reduce face count after generation. Recommended — raw meshes can have 200k+ faces. |
| `max_facenum` | INT | `40000` | Maximum faces when simplify is on. |
| `remove_background` | BOOLEAN | `False` | Run background removal on each image before generation. Requires the `rembg` pip package. Usually better to prepare images beforehand. |
| `attention_mode` | ENUM | `"sdpa"` | sdpa = standard, sageattn = SageAttention. |
| `skip_generated_mesh` | BOOLEAN | `True` | Skip images that already have a GLB in the output folder. Enables resumable batch processing. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `input_folder` | STRING | Pass-through of the input folder path. |
| `output_folder` | STRING | Pass-through of the output folder path. |
| `processed_images` | STRING | List of input images that were processed. |
| `processed_meshes` | STRING | List of generated GLB file paths. |
