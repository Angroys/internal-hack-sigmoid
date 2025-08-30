# -*- coding: utf-8 -*-
"""
Diffusers-based generator for a photorealistic cyberpunk futuristic city using:
- Base model: stable-diffusion-v1-5
- LoRA: Remade-AI/Cyberpunk
- ControlNet: lllyasviel/sd-controlnet-depth
- IP-Adapter: h94/IP-Adapter

This code follows OOP principles and is written to be clear and extensible.

Notes:
- Requires diffusers, transformers, accelerate, safetensors, torch, PIL
- Some model integrations (LoRA, IP-Adapter) rely on newer diffusers helper methods
  (e.g. `.unet.load_attn_procs`). The code tries safe fallbacks.

Usage:
- Instantiate DiffusionEngine and call generate().
"""

from typing import Optional, Union
import os
import torch
from PIL import Image
import numpy as np

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

# Optional import for IP-Adapter; if not available, the code will skip attaching it.
try:
    from diffusers import IPAdapterModel  # type: ignore

    _HAS_IP_ADAPTER = True
except Exception:
    _HAS_IP_ADAPTER = False


class DiffusionEngine:
    def __init__(
        self,
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        lora: Optional[str] = "Remade-AI/Cyberpunk",
        controlnet: Optional[str] = "lllyasviel/sd-controlnet-depth",
        ip_adapter: Optional[str] = "h94/IP-Adapter",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.base_model = base_model
        self.lora = lora
        self.controlnet = controlnet
        self.ip_adapter = ip_adapter if _HAS_IP_ADAPTER else None
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Prefer float16 on CUDA for memory/perf, fallback to float32
        self.torch_dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        self.pipeline: Optional[StableDiffusionControlNetPipeline] = None
        self._load_models()

    def _load_models(self) -> None:
        """Load ControlNet and Stable Diffusion pipeline, then attach LoRA and IP-Adapter if possible."""
        print(f"Loading ControlNet from: {self.controlnet}")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet, torch_dtype=self.torch_dtype
        )

        print(f"Loading base Stable Diffusion pipeline from: {self.base_model}")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=self.torch_dtype,
            safety_checker=None,  # optional; disable if you don't want safety filter
        )

        # Use a deterministic scheduler good for photorealistic results
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Move pipe to device
        pipe = pipe.to(self.device)

        # Try to load LoRA onto the UNet (newer diffusers supports load_attn_procs)
        if self.lora:
            try:
                print(f"Attempting to load LoRA from: {self.lora}")
                # Preferred API (diffusers>=0.14): load attention processors
                pipe.unet.load_attn_procs(self.lora)
                print("LoRA loaded into UNet via attn_procs.")
            except Exception as e:
                print(
                    "Could not load LoRA via load_attn_procs. Attempting fallback. Error:",
                    e,
                )
                try:
                    # Fallback: try to load lora weights into the whole pipeline if available
                    pipe.load_lora_weights(self.lora)
                    print("LoRA loaded via pipe.load_lora_weights fallback.")
                except Exception as e2:
                    print("Fallback LoRA loading failed:", e2)
                    print(
                        "Continuing without LoRA. If you want LoRA, ensure it's available in attn_procs format or supported by your diffusers version."
                    )

        # Try to attach IP-Adapter if available and requested
        if self.ip_adapter:
            if _HAS_IP_ADAPTER:
                try:
                    print(f"Loading IP-Adapter from: {self.ip_adapter}")
                    ip_adapter_model = IPAdapterModel.from_pretrained(
                        self.ip_adapter, torch_dtype=self.torch_dtype
                    )
                    # Attach to the pipeline in a conventional attribute name; some pipelines expect an adapter attr.
                    pipe.ip_adapter = ip_adapter_model.to(self.device)
                    print("IP-Adapter attached to pipeline.")
                except Exception as e:
                    print("Failed to load IP-Adapter:", e)
                    print("Continuing without IP-Adapter.")
            else:
                print("IP-Adapter support not installed; skipping IP-Adapter.")

        # Store the pipeline
        self.pipeline = pipe

    @staticmethod
    def _prepare_image(img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Ensure the input is a PIL RGB image."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        seed: Optional[int] = None,
        depth_image: Optional[Union[Image.Image, np.ndarray]] = None,
        ip_adapter_image: Optional[Union[Image.Image, np.ndarray]] = None,
    ) -> Image.Image:
        """
        Generate an image.
        - depth_image: used as the ControlNet conditioning map (depth map)
        - ip_adapter_image: used as IP-Adapter conditioning (reference/style)

        Returns a PIL Image
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded.")

        pipe = self.pipeline

        # Prepare RNG
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        # Prepare conditioning images
        controlnet_image = None
        if depth_image is not None:
            controlnet_image = self._prepare_image(depth_image)

        adapter_image = None
        if (
            ip_adapter_image is not None
            and getattr(pipe, "ip_adapter", None) is not None
        ):
            adapter_image = self._prepare_image(ip_adapter_image)

        # Compose call kwargs dynamically to be robust across diffusers versions
        call_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt

        # Attach ControlNet conditioning image if provided
        if controlnet_image is not None:
            # StableDiffusionControlNetPipeline expects `image` for conditioning maps
            call_kwargs["image"] = controlnet_image

        # Attach IP-Adapter conditioning if present (the exact kwarg name can vary with versions)
        if adapter_image is not None:
            # Many integrations accept `adapter_image` or `image_adapter`
            # Try both in order of common usage
            try:
                call_kwargs["adapter_image"] = adapter_image
            except Exception:
                call_kwargs["image_adapter"] = adapter_image

        # Call pipeline
        print("Generating image with prompt:\n", prompt)
        out = pipe(**call_kwargs)

        # `out` is usually a StableDiffusionPipelineOutput with `.images`
        images = getattr(out, "images", None) or out
        if isinstance(images, list):
            image = images[0]
        elif isinstance(images, Image.Image):
            image = images
        else:
            # try to convert numpy
            image = Image.fromarray(np.asarray(images))

        return image


if __name__ == "__main__":
    # Example usage
    engine = DiffusionEngine()

    prompt = (
        "A photorealistic futuristic city skyline at night in cyberpunk style, towering skyscrapers, neon signs, "
        "wet streets reflecting lights, volumetric fog, intricate architectural detail, dramatic cinematic lighting"
    )
    negative = "lowres, text, watermark, deformed, bad anatomy"

    # If you have a depth map (grayscale) for camera framing, load it here; otherwise leave None
    depth_map = None
    # If you have a reference image for IP-Adapter (composition/style guidance), load it here; otherwise None
    ip_ref = None

    result = engine.generate(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=28,
        guidance_scale=7.5,
        height=1024,
        width=768,
        seed=42,
        depth_image=depth_map,
        ip_adapter_image=ip_ref,
    )

    # Save output
    output_path = "cyberpunk_city.png"
    result.save(output_path)
    print(f"Saved image to {output_path}")
