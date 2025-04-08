# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import ipdb
import argparse
import os

import torch
from PIL import Image

from pipelines.pipeline_infu_flux_WithouConrt import InfUFluxPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_image', default='/SAN/intelsys/IdPreservProject/epic/dataset/FormatPhoto/P6.png',
                        help="""input ID image""")
    parser.add_argument('--out_results_dir', default='./results', help="""output folder""")
    parser.add_argument('--prompt', default='A man, portrait, cinematic')
    parser.add_argument('--base_model_path', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_dir', default='/SAN/intelsys/IdPreservProject/epic/models/InfiniteYou')
    parser.add_argument('--infu_flux_version', default='v1.0',
                        help="""InfiniteYou-FLUX version: currently only v1.0""")
    parser.add_argument('--model_version', default='aes_stage2',
                        help="""model version: aes_stage2 | sim_stage1""")
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int, help="""seed (0 for random)""")
    parser.add_argument('--guidance_scale', default=3.5, type=float)
    parser.add_argument('--num_steps', default=30, type=int)
    parser.add_argument('--infusenet_conditioning_scale', default=1.0, type=float)
    parser.add_argument('--infusenet_guidance_start', default=0.0, type=float)
    parser.add_argument('--infusenet_guidance_end', default=1.0, type=float)
    parser.add_argument('--emotion', default="amusement")

    args = parser.parse_args()

    # Check arguments
    assert args.infu_flux_version == 'v1.0', 'Currently only supports InfiniteYou-FLUX v1.0'
    assert args.model_version in ['aes_stage2', 'sim_stage1'], \
        'Currently only supports model versions: aes_stage2 | sim_stage1'

    # Set cuda device
    torch.cuda.set_device(args.cuda_device)

    # Load pipeline
    infu_model_path = os.path.join(args.model_dir, f'infu_flux_{args.infu_flux_version}', args.model_version)
    insightface_root_path = os.path.join(args.model_dir, 'supports', 'insightface')
    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
    )

    # Perform inference
    if args.seed == 0:
        args.seed = torch.seed() & 0xFFFFFFFF
    image = pipe(
        id_image=Image.open(args.id_image).convert('RGB'),
        prompt=args.prompt,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        infusenet_conditioning_scale=args.infusenet_conditioning_scale,
        infusenet_guidance_start=args.infusenet_guidance_start,
        infusenet_guidance_end=args.infusenet_guidance_end,
    )

    # Save results
    os.makedirs(args.out_results_dir, exist_ok=True)
    out_result_path = os.path.join(args.out_results_dir, f'{args.emotion}.png')
    image.save(out_result_path)


if __name__ == "__main__":
    # ipdb.set_trace()
    main()
# python InfiniteYou/test_WithouConrt.py --emotion "sadness" --id_image epic/dataset/FormatPhoto/P6.png --out_results_dir ./results --prompt "A somber man sits on a weathered bench in a misty park. His head bowed, hands clasped tightly, tears glisten under a muted, overcast sky. In the style of Romanticism, soft, melancholic colors envelop him, the background a blur of drooping willows and a distant, gentle rain, enhancing the scene's poignant atmosphere."
