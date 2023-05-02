import gc
from typing import *

import gradio as gr
import numpy as np
import PIL.Image
import torch
from diffusers import ControlNetModel
from diffusers.utils import PIL_INTERPOLATION

from api.events import event_handler
from api.events.common import PreUICreateEvent
from api.events.generation import LoadResourceEvent, UNetDenoisingEvent
from api.models.diffusion import ImageGenerationOptions
from api.plugin import get_plugin_id
from modules import config
from modules.logger import logger
from modules.plugin_loader import register_plugin_ui
from modules.shared import hf_diffusers_cache_dir

from . import preprocessors

plugin_id = get_plugin_id()
controlnet_model: Optional[ControlNetModel] = None
available_preprocessors = ["canny", "openpose"]
available_controlnet_models = [
    "lllyasviel/sd-controlnet-canny",
    "lllyasviel/sd-controlnet-mlsd",
    "lllyasviel/sd-controlnet-seg",
    "lllyasviel/sd-controlnet-hed",
    "lllyasviel/sd-controlnet-normal",
    "lllyasviel/sd-controlnet-scribble",
    "lllyasviel/sd-controlnet-depth",
    "lllyasviel/sd-controlnet-openpose",
]


def preprocess_image(image: PIL.Image, preprocessor: str):
    if hasattr(preprocessors, preprocessor):
        return getattr(preprocessors, preprocessor)(image)
    else:
        logger.warning(f"Preprocessor {preprocessor} not found")
        return image


def ui():
    with gr.Column():
        with gr.Row():
            image = gr.Image(label="Input Image", type="pil")
            preview_image = gr.Image(label="Preview", interactive=False, visible=False)
        with gr.Row():
            enabled = gr.Checkbox(label="Enable", value=False)

        with gr.Row():
            preprocessor = gr.Dropdown(
                label="Preprocessor",
                choices=["none"] + available_preprocessors,
                value="none",
            )
            controlnet_model_id = gr.Dropdown(
                label="ControlNet model",
                choices=available_controlnet_models,
                value="lllyasviel/sd-controlnet-canny",
            )
        with gr.Row():
            control_weight = gr.Slider(
                label="Control weight",
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                value=1.0,
            )
            start_control_step = gr.Slider(
                label="Start control step",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
            )
            end_control_step = gr.Slider(
                label="End control step",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
            )

        with gr.Row():
            guess_mode = gr.Checkbox(label="Guess mode", value=False)
        with gr.Row():
            preview = gr.Button("Preview")

    def run_preview(image, preprocessor):
        return gr.Image.update(
            visible=True, value=preprocess_image(image, preprocessor)
        )

    preview.click(run_preview, inputs=[image, preprocessor], outputs=[preview_image])

    return [
        image,
        enabled,
        preprocessor,
        controlnet_model_id,
        control_weight,
        start_control_step,
        end_control_step,
        guess_mode,
    ]


@event_handler()
def pre_ui_create(e: PreUICreateEvent):
    register_plugin_ui(ui)


def prepare_image(
    image: PIL.Image.Image,
    width: int,
    height: int,
    batch_size: int,
    num_images_per_prompt: int,
    device: torch.device,
    dtype: torch.dtype,
    do_classifier_free_guidance: bool = False,
    guess_mode: bool = False,
):
    if not isinstance(image, torch.Tensor):
        if isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            images = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = image_.resize(
                    (width, height), resample=PIL_INTERPOLATION["lanczos"]
                )
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)

            image = images

            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image


@event_handler()
def load_resource(e: LoadResourceEvent):
    global controlnet_model
    (
        image,
        enabled,
        preprocessor,
        controlnet_model_id,
        control_weight,
        start_control_step,
        end_control_step,
        guess_mode,
    ) = e.pipe.plugin_data[plugin_id]
    if not enabled:
        if controlnet_model is not None:
            del controlnet_model
            torch.cuda.empty_cache()
            gc.collect()
            controlnet_model = None
        return

    if type(e.pipe).__mode__ != "diffusers":
        logger.warning("ControlNet plugin is only available in diffusers mode")
        e.pipe.plugin_data[plugin_id][1] = False
        return

    if preprocessor != "none":
        image = preprocess_image(image, preprocessor)

    opts: ImageGenerationOptions = e.pipe.opts
    controlnet_model = ControlNetModel.from_pretrained(
        controlnet_model_id,
        use_auth_token=config.get("hf_token"),
        cache_dir=hf_diffusers_cache_dir(),
        torch_dtype=torch.float16,
    ).to(e.pipe.device)
    e.pipe.plugin_data[plugin_id][0] = prepare_image(
        image,
        opts.width,
        opts.height,
        opts.batch_size,
        1,
        e.pipe.device,
        controlnet_model.dtype,
        do_classifier_free_guidance=opts.guidance_scale > 1.0,
        guess_mode=guess_mode,
    )


@event_handler()
def pre_unet_predict(e: UNetDenoisingEvent):
    (
        image,
        enabled,
        _,
        _,
        control_weight,
        start_control_step,
        end_control_step,
        guess_mode,
    ) = e.pipe.plugin_data[plugin_id]
    opts: ImageGenerationOptions = e.pipe.opts
    if (
        not enabled
        or e.step / opts.num_inference_steps < start_control_step
        or e.step / opts.num_inference_steps > end_control_step
    ):
        return

    if guess_mode and e.do_classifier_free_guidance:
        # Infer ControlNet only for the conditional batch.
        controlnet_latent_model_input = e.latents
        controlnet_prompt_embeds = e.prompt_embeds.chunk(2)[1]
    else:
        controlnet_latent_model_input = e.latent_model_input
        controlnet_prompt_embeds = e.prompt_embeds

    down_block_res_samples, mid_block_res_sample = controlnet_model(
        controlnet_latent_model_input,
        e.timestep,
        encoder_hidden_states=controlnet_prompt_embeds,
        controlnet_cond=image,
        conditioning_scale=control_weight,
        guess_mode=guess_mode,
        return_dict=False,
    )

    if guess_mode and e.do_classifier_free_guidance:
        # Infered ControlNet only for the conditional batch.
        # To apply the output of ControlNet to both the unconditional and conditional batches,
        # add 0 to the unconditional batch to keep it unchanged.
        down_block_res_samples = [
            torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
        ]
        mid_block_res_sample = torch.cat(
            [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
        )

    e.unet_additional_kwargs = {
        "down_block_additional_residuals": down_block_res_samples,
        "mid_block_additional_residual": mid_block_res_sample,
    }
