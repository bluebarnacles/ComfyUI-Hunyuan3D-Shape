# ComfyUI-Hunyuan3d-Shape

ComfyUI nodes for generating 3D meshes from images using Tencent's Hunyuan3D-2.1 DiT model. Powered by Tencent Hunyuan.

This is the **shape generation** half of the Hunyuan3D pipeline. For texture painting, see [ComfyUI-Hunyuan3d-Paint](https://github.com/agenticvibes/ComfyUI-Hunyuan3d-Paint).

## Nodes

| Node | Description |
|---|---|
| **Mesh Generator** | Generate 3D mesh latents from an image using DiT flow-matching diffusion |
| **VAE Loader** | Load a ShapeVAE checkpoint |
| **VAE Config** | Override ShapeVAE parameters |
| **VAE Decode** | Decode latents to 3D mesh via marching cubes |
| **Resize Images** | Resize images with configurable interpolation |
| **Load Image with Transparency** | Load image and extract alpha channel as mask |
| **Mesh Generation Batch** | Batch process a folder of images to meshes |

## Typical Workflow

```
Image → Mesh Generator → VAE Decode → TRIMESH output
```

The output `TRIMESH` can be piped to [ComfyUI-MeshTools](https://github.com/agenticvibes/ComfyUI-MeshTools) for post-processing, UV unwrapping, and export — or to [ComfyUI-Hunyuan3d-Paint](https://github.com/agenticvibes/ComfyUI-Hunyuan3d-Paint) for texture generation.

## Installation

### 1. Clone or copy into ComfyUI

```
ComfyUI/custom_nodes/ComfyUI-Hunyuan3d-Shape/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download model checkpoints

Download from [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) and place in your ComfyUI models directory:

```
ComfyUI/models/
├── diffusion_models/
│   └── hunyuan3d-dit-v2-1.ckpt       ← DiT model (~7.4 GB)
└── vae/
    └── hunyuan3d-vae-v2-1.ckpt       ← ShapeVAE (~656 MB)
```

**Direct download links:**
- [hunyuan3d-dit-v2-1](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-dit-v2-1)
- [hunyuan3d-vae-v2-1](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-vae-v2-1)

## License

This project contains code from Tencent's Hunyuan3D-2.1, licensed under the [Tencent Hunyuan 3D 2.1 Community License Agreement](LICENSE).

**Important restrictions:**
- **Non-commercial use only**
- **Territory restricted** — excludes European Union, United Kingdom, and South Korea
- See [LICENSE](LICENSE) for full terms

## Acknowledgements

- [Tencent](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — Hunyuan3D-2.1 model and original code
- [visualbruno](https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1) — Original ComfyUI wrapper
- [kijai](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper) — Original Hunyuan3D v2.0 wrapper
