# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# ...
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math
import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers.models import FluxControlNetModel
from facexlib.recognition import init_recognition_model
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image

from .pipeline_flux_infusenet_WithouConrt import FluxInfuseNetPipeline
from .resampler import Resampler


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_arcface_bgr_embedding(in_image, landmark, arcface_model=None):
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(landmark), image_size=112)
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    arc_face_image = 2.0 * arc_face_image - 1.0
    arc_face_image = arc_face_image.cuda().contiguous()
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    face_emb = arcface_model(arc_face_image)[0]  # [512], normalized
    return face_emb


def resize_and_pad_image(source_img, target_img_size):
    """
    如果只想保留纯人脸检测部分，可以去掉此函数。
    此处保留是为了说明若还需对 ID 图像做其他 resize/pad，可自定义处理。
    """
    source_img_size = source_img.size
    target_width, target_height = target_img_size
    
    if target_width <= target_height:
        new_width = target_width
        new_height = int(target_width * (source_img_size[1] / source_img_size[0]))
    else:
        new_height = target_height
        new_width = int(target_height * (source_img_size[0] / source_img_size[1]))
    
    resized_source_img = source_img.resize((new_width, new_height), Image.LANCZOS)
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    padded_img = Image.new("RGB", target_img_size, (255, 255, 255))
    padded_img.paste(resized_source_img, (pad_left, pad_top))
    return padded_img


class InfUFluxPipeline:
    def __init__(
        self, 
        base_model_path, 
        infu_model_path, 
        insightface_root_path='./',
        image_proj_num_tokens=8,
        infu_flux_version='v1.0',
        model_version='aes_stage2',
    ):
        self.infu_flux_version = infu_flux_version
        self.model_version = model_version
        
        # 如果本地没有对应模型，会尝试从 HuggingFace 下载
        try:
            infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
            self.infusenet = FluxControlNetModel.from_pretrained(
                infusenet_path, torch_dtype=torch.bfloat16
            )
        except:
            print("No InfiniteYou model found. Downloading from HuggingFace ...")
            snapshot_download(
                repo_id='ByteDance/InfiniteYou',
                local_dir='./models/InfiniteYou',
                local_dir_use_symlinks=False
            )
            infu_model_path = os.path.join('./models/InfiniteYou', f'infu_flux_{infu_flux_version}', model_version)
            infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
            self.infusenet = FluxControlNetModel.from_pretrained(
                infusenet_path, torch_dtype=torch.bfloat16
            )
            insightface_root_path = './models/InfiniteYou/supports/insightface'

        try:
            pipe = FluxInfuseNetPipeline.from_pretrained(
                base_model_path,
                controlnet=self.infusenet,
                torch_dtype=torch.bfloat16,
            )
        except:
            try:
                pipe = FluxInfuseNetPipeline.from_single_file(
                    base_model_path,
                    controlnet=self.infusenet,
                    torch_dtype=torch.bfloat16,
                )
            except Exception as e:
                print(e)
                exit()

        pipe.to('cuda', torch.bfloat16)
        self.pipe = pipe

        # Load image_proj_model
        num_tokens = image_proj_num_tokens
        image_emb_dim = 512
        self.image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        image_proj_model_path = os.path.join(infu_model_path, 'image_proj_model.bin')
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        self.image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
        self.image_proj_model.to('cuda', torch.bfloat16)
        self.image_proj_model.eval()

        # Load face encoders
        print("now is loading face encoder")
        self.app_640 = FaceAnalysis(
            name='antelopev2', root=insightface_root_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))

        self.app_320 = FaceAnalysis(
            name='antelopev2', root=insightface_root_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))

        self.app_160 = FaceAnalysis(
            name='antelopev2', root=insightface_root_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))

        self.arcface_model = init_recognition_model('arcface', device='cuda')

    def load_loras(self, loras):
        names, scales = [], []
        for lora_path, lora_name, lora_scale in loras:
            if lora_path != "":
                print(f"loading lora {lora_path}")
                self.pipe.load_lora_weights(lora_path, adapter_name=lora_name)
                names.append(lora_name)
                scales.append(lora_scale)
        if len(names) > 0:
            self.pipe.set_adapters(names, adapter_weights=scales)

    def _detect_face(self, id_image_cv2):
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info

        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info

        face_info = self.app_160.get(id_image_cv2)
        return face_info

    def __call__(
        self,
        id_image: Image.Image,
        prompt: str,
        width=864,
        height=1152,
        seed=42,
        guidance_scale=3.5,
        num_steps=30,
        infusenet_conditioning_scale=1.0,
        infusenet_guidance_start=0.0,
        infusenet_guidance_end=1.0,
    ):
        # Extract ID embeddings
        print('Preparing ID embeddings')
        id_image_cv2 = cv2.cvtColor(np.array(id_image), cv2.COLOR_RGB2BGR)
        face_info = self._detect_face(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')

        face_info = sorted(
            face_info,
            key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
        )[-1]
        landmark = face_info['kps']
        id_embed = extract_arcface_bgr_embedding(id_image_cv2, landmark, self.arcface_model)
        id_embed = id_embed.clone().unsqueeze(0).float().cuda()
        id_embed = id_embed.reshape([1, -1, 512]).to(device='cuda', dtype=torch.bfloat16)

        with torch.no_grad():
            id_embed = self.image_proj_model(id_embed)
            bs_embed, seq_len, _ = id_embed.shape
            id_embed = id_embed.repeat(1, 1, 1).view(bs_embed, seq_len, -1).to(device='cuda', dtype=torch.bfloat16)

        # 不需要 control_image 的逻辑，直接传入 None 或不传
        # 设置随机种子
        seed_everything(seed)

        print('Generating image')
        image = self.pipe(
            prompt=prompt,
            controlnet_prompt_embeds=id_embed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            controlnet_guidance_scale=1.0,
            controlnet_conditioning_scale=infusenet_conditioning_scale,
            control_guidance_start=infusenet_guidance_start,
            control_guidance_end=infusenet_guidance_end,
            height=height,
            width=width,
        ).images[0]

        return image
