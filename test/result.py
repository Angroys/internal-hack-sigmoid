"""
SDXL oil-painting + digital-art generator using Diffusers

- Uses:
  - Base model: stabilityai/stable-diffusion-xl-base-1.0
  - LoRA: ntc-ai/SDXL-LoRA-slider.epic-oil-painting
  - ControlNet (Canny): lllyasviel/sd-controlnet-canny
  - IP Adapter: InvokeAI/ip_adapter_sdxl

Design:
- Object-oriented, single class SDXLGenerator with load and generate methods
- Uses .load_attn_procs to apply LoRA to the pipeline's UNet
- Uses ControlNetModel and IPAdapterModel and integrates them into the pipeline

Notes:
- This is a template. API names for some classes/arguments may vary by diffusers/versions.
- Ensure diffusers, transformers, accelerate, safetensors and xformers (optional) are installed.

"""

import os
from typing import Optional

import torch
from PIL import Image

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    IPAdapterModel,
    UniPCMultistepScheduler,
)


class SDXLGenerator:
    """Generator that composes SDXL base + LoRA + ControlNet(Canny) + IP Adapter.

    Example:
        gen = SDXLGenerator(device='cuda', dtype=torch.float16)
        gen.load_models()
        img = gen.generate(
            prompt="A dramatic oil painting merged with neon-lit digital art, cinematic lighting",
            negative_prompt="lowres, bad anatomy",
            control_image_path="control_canny.png",
            ip_adapter_image_path="style_reference.jpg",
            seed=42,
        )
        img.save("output.png")
    """

    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_repo: str = "ntc-ai/SDXL-LoRA-slider.epic-oil-painting",
        controlnet_repo: str = "lllyasviel/sd-controlnet-canny",
        ip_adapter_repo: str = "InvokeAI/ip_adapter_sdxl",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_xformers: bool = True,
    ):
        self.base_model = base_model
        self.lora_repo = lora_repo
        self.controlnet_repo = controlnet_repo
        self.ip_adapter_repo = ip_adapter_repo
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.dtype = dtype
        self.enable_xformers = enable_xformers

        # will be populated on load
        self.pipeline: Optional[StableDiffusionXLControlNetPipeline] = None

    def load_models(self):
        """Download and assemble the pipeline components.

        This performs the following:
        - loads ControlNetModel
        - loads IPAdapterModel
        - loads SDXL pipeline with the ControlNet and IPAdapter plugged in
        - applies LoRA weights into the UNet with load_attn_procs
        - moves pipeline to target device and dtype
        """
        # Load ControlNet (Canny)
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_repo,
            torch_dtype=self.dtype,
            # trust repo for custom modules if necessary
            local_files_only=False,
        )

        # Load IP Adapter (for image-conditioned guidance)
        ip_adapter = IPAdapterModel.from_pretrained(
            self.ip_adapter_repo,
            torch_dtype=self.dtype,
            local_files_only=False,
        )

        # Load main pipeline with ControlNet and IP Adapter integrated
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            ip_adapter=ip_adapter,
            torch_dtype=self.dtype,
            local_files_only=False,
        )

        # Update scheduler to a fast/stable multistep scheduler (recommended for SDXL)
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

        # Apply LoRA weights into the UNet (attn processors)
        # This will load LoRA-attn-proc-style weights that modify the attention layers.
        try:
            pipeline.unet.load_attn_procs(self.lora_repo)
        except Exception as exc:
            # If the repo contains a file rather than remote repo, attempt direct path
            print(f"Warning: load_attn_procs failed. Exception: {exc}")
            print(
                "Make sure the LoRA repo or file is accessible and compatible with diffusers' load_attn_procs."
            )

        # Optional: enable xformers for memory-efficient attention (if installed)
        if self.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                # xformers may not be installed; continue without it
                pass

        # Move pipeline to device
        pipeline = pipeline.to(self.device)

        self.pipeline = pipeline

    @staticmethod
    def _load_image(image_path: str, size: Optional[tuple] = None) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        if size:
            img = img.resize(size, resample=Image.LANCZOS)
        return img

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        control_image_path: Optional[str] = None,
        ip_adapter_image_path: Optional[str] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        height: int = 1024,
        width: int = 1024,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """Run a generation with the assembled pipeline.

        - control_image_path: path to a Canny edge image (grayscale or RGB) for ControlNet conditioning
        - ip_adapter_image_path: reference image for the IP Adapter (style/content)

        Returns the generated PIL.Image. If output_path is provided, image is saved to that path.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_models() first.")

        # Prepare conditioning images
        control_image = None
        if control_image_path:
            control_image = self._load_image(control_image_path, size=(width, height))

        ip_adapter_image = None
        if ip_adapter_image_path:
            ip_adapter_image = self._load_image(
                ip_adapter_image_path, size=(width // 2, height // 2)
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run pipeline. Parameter names may vary depending on diffusers version; common ones are:
        # prompt, negative_prompt, image / control_image / controlnet_conditioning_image, ip_adapter_image
        # We'll try a few commonly accepted keywords for compatibility.
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

        # attach control image under likely accepted names
        if control_image is not None:
            # try 'image' first (many pipelines accept this)
            call_kwargs["image"] = control_image
            # also set 'control_image' for versions that expect that
            call_kwargs.setdefault("control_image", control_image)
            # also include controlnet conditioning alias
            call_kwargs.setdefault("controlnet_conditioning_image", control_image)

        # attach ip adapter image under the key the pipeline expects
        if ip_adapter_image is not None:
            call_kwargs["ip_adapter_image"] = ip_adapter_image
            call_kwargs.setdefault("image_adapter_image", ip_adapter_image)

        # Execute
        output = self.pipeline(**call_kwargs)

        # pipeline outputs may include .images
        result_img = None
        if hasattr(output, "images") and len(output.images) > 0:
            result_img = output.images[0]
        elif isinstance(output, Image.Image):
            result_img = output
        else:
            # try to extract from dict-like output
            try:
                result_img = output["images"][0]
            except Exception:
                raise RuntimeError("Could not extract image from pipeline output.")

        if output_path:
            result_img.save(output_path)

        return result_img


if __name__ == "__main__":
    # Example usage - edit paths and prompt as desired
    gen = SDXLGenerator(device="cuda", dtype=torch.float16, enable_xformers=True)
    gen.load_models()

    prompt = (
        "A vibrant oil painting fused with cyber-digital art: dramatic brush strokes, rich impasto textures, "
        "neon rim lighting, high detail, cinematic composition"
    )

    negative = "lowres, deformed, bad anatomy, text"

    control_path = os.path.join(
        "./resources", "control_canny.png"
    )  # provide a Canny edge image
    ip_adapter_path = os.path.join(
        "./resources", "style_reference.jpg"
    )  # optional style/reference

    out = gen.generate(
        prompt=prompt,
        negative_prompt=negative,
        control_image_path=control_path,
        ip_adapter_image_path=ip_adapter_path,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=1234,
        height=1024,
        width=1024,
        output_path="./output_oil_digital.png",
    )

    print("Saved generated image to ./output_oil_digital.png")
