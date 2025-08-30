"""
SDXL Oil Painting + Digital Art generator using Diffusers
- Base model: stabilityai/stable-diffusion-xl-base-1.0
- LoRA: Eunju2834/LoRA_oilcanvas_style
- IP Adapter: XLabs-AI/flux-ip-adapter

Notes:
- Requires diffusers, transformers, accelerate, torch. For LoRA support you may need diffusers >= 0.19 and for some helpers 'lora-diffusion'.
- This code tries multiple integration methods and prints actionable messages on fallback.

Usage example:
    gen = SDXLOilDigitalGenerator(hf_token="<HF_TOKEN>")
    gen.load_components()
    imgs = gen.generate("A serene oil painting fused with vibrant digital art, dramatic lighting", num_images=2)
    for i, im in enumerate(imgs):
        im.save(f"oil_digital_{i}.png")

Return: list of PIL images
"""

import os
import torch
from typing import List, Optional

from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)


class SDXLOilDigitalGenerator:
    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora: Optional[str] = "Eunju2834/LoRA_oilcanvas_style",
        ip_adapter: Optional[str] = "XLabs-AI/flux-ip-adapter",
        hf_token: Optional[str] = None,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.base_model = base_model
        self.lora = lora
        self.ip_adapter = ip_adapter
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        # Will be initialized by load_components()
        self.pipe: Optional[StableDiffusionXLPipeline] = None

    def load_components(self):
        """Load the SDXL pipeline, attach LoRA and IP adapter where possible.
        This method keeps operations modular and reports helpful diagnostics.
        """
        print("Loading base SDXL pipeline...")
        torch_dtype = torch.float16 if self.use_fp16 else torch.float32

        # Load base pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model,
            use_auth_token=self.hf_token,
            torch_dtype=torch_dtype,
        )

        # Replace scheduler with DPMSolverMultistep for faster convergence (optional)
        try:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
        except Exception:
            # If replacement fails, keep the original scheduler
            pass

        # Performance helpers
        try:
            # enable memory efficient attention if available
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        self.pipe = self.pipe.to(self.device)
        print(f"Pipeline loaded on device: {self.device} (dtype={torch_dtype})")

        # Apply LoRA if provided
        if self.lora:
            self._apply_lora(self.lora)

        # Attach IP adapter if provided
        if self.ip_adapter:
            self._attach_ip_adapter(self.ip_adapter)

    def _apply_lora(self, lora_id: str):
        """Try multiple strategies to apply LoRA weights to the pipeline.
        Preferred: pipe.load_lora_weights (newer diffusers). Fallback to lora-diffusion injection.
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded. Call load_components() first.")

        print(f"Applying LoRA weights: {lora_id}")
        # Strategy 1: diffusers built-in helper
        try:
            # Newer diffusers exposes load_lora_weights on pipelines
            load_lora = getattr(self.pipe, "load_lora_weights", None)
            if callable(load_lora):
                load_lora(lora_id, weight_name=None, device=self.device)
                print("Loaded LoRA via pipeline.load_lora_weights()")
                return
        except Exception as e:
            print("pipeline.load_lora_weights failed:", e)

        # Strategy 2: use lora-diffusion library to inject into the UNet (if installed)
        try:
            from lora_diffusion import inject_lora_weights

            # inject into unet; inject_lora_weights typically accepts (unet, lora_weights_or_hf_id)
            inject_lora_weights(self.pipe.unet, lora_id)
            print("Injected LoRA into UNet using lora-diffusion")
            return
        except Exception as e:
            print("lora-diffusion injection failed or not installed:", e)

        # Strategy 3: Ask user to manually merge or ensure compatibility
        print(
            "Could not automatically apply the LoRA.\n"
            "Please ensure you have a compatible diffusers version or install 'lora-diffusion'.\n"
            "If the LoRA is in safetensors format, consider merging weights offline or using the official loader API."
        )

    def _attach_ip_adapter(self, ip_adapter_id: str):
        """Attempt to load and attach an IP Adapter for SDXL.
        Implementation varies across diffusers versions; try a few approaches and warn on failure.
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded. Call load_components() first.")

        print(f"Attaching IP Adapter: {ip_adapter_id}")

        # Strategy 1: diffusers IPAdapterModel + pipeline registration (newer diffusers)
        try:
            from diffusers import IPAdapterModel

            ip_model = IPAdapterModel.from_pretrained(
                ip_adapter_id,
                use_auth_token=self.hf_token,
                torch_dtype=(torch.float16 if self.use_fp16 else torch.float32),
            )

            # Some diffusers versions expose a registration function
            register_fn = getattr(self.pipe, "register_to_ip_adapter", None)
            if callable(register_fn):
                register_fn(ip_model)
                print("IP Adapter registered via pipeline.register_to_ip_adapter()")
                return

            # Otherwise bind as attribute and hope for pipeline support during call
            setattr(self.pipe, "ip_adapter", ip_model)
            print(
                "Attached IP Adapter to pipeline as '.ip_adapter' attribute (verify your diffusers version)."
            )
            return
        except Exception as e:
            print(
                "Automatic IP Adapter attachment failed or unsupported in this diffusers version:",
                e,
            )

        print(
            "Could not attach IP Adapter automatically.\n"
            "Ensure you are using a diffusers version with IP-Adapter support for SDXL, or consult the adapter's README for manual integration steps."
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        save_to: Optional[str] = None,
    ) -> List:
        """Generate images from prompt. Returns list of PIL.Image objects.

        If an IP adapter is attached and expects an image input, this method does not pass any image
        by default. To use ip-adapter conditioned on an example image, extend this wrapper to provide
        an 'ip_adapter_image' argument to the underlying pipeline (API varies across diffusers releases).
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded. Call load_components() first.")

        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        # The exact call signature differs by diffusers release. We call common arguments.
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "num_images_per_prompt": num_images,
            "generator": generator,
        }

        # Filter out None values (e.g., negative_prompt)
        call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

        print("Generating images...")
        with torch.autocast(self.device.type if torch.cuda.is_available() else "cpu"):
            output = self.pipe(**call_kwargs)

        images = getattr(output, "images", output)

        if save_to:
            os.makedirs(save_to, exist_ok=True)
            for i, img in enumerate(images):
                path = os.path.join(save_to, f"generated_{i}.png")
                img.save(path)
            print(f"Saved {len(images)} images to {save_to}")

        return images


if __name__ == "__main__":
    # Example quick-run (replace HF token with your token or set HF_TOKEN env var)
    HF_TOKEN = os.getenv("HF_TOKEN")
    gen = SDXLOilDigitalGenerator(hf_token=HF_TOKEN)
    gen.load_components()

    prompt = (
        "A fusion of classical oil painting textures with vibrant digital art elements, "
        "rich impasto brushstrokes, dramatic lighting, cinematic composition, high detail"
    )

    imgs = gen.generate(
        prompt, num_images=1, seed=42, num_inference_steps=28, guidance_scale=4.5
    )
    for idx, im in enumerate(imgs):
        im.save(f"oil_digital_result_{idx}.png")
    print("Done.")
