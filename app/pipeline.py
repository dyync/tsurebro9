import os, numpy as np, torch, imageio, shutil
from PIL import Image
from typing import List, Tuple, Literal
from easydict import EasyDict as edict
from .render_utils import render_video
from trellis import TrellisImageTo3DPipeline, Gaussian, MeshExtractResult

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large").cuda()

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }

def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    session_hash: str,
    **kwargs
) -> dict:
    user_dir = os.path.join(TMP_DIR, str(session_hash))
    os.makedirs(user_dir, exist_ok=True)

    if not is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={"steps": ss_sampling_steps, "cfg_strength": ss_guidance_strength},
            slat_sampler_params={"steps": slat_sampling_steps, "cfg_strength": slat_guidance_strength},
        )
    else:
        outputs = pipeline.run_multi_image(
            [img[0] for img in multiimages],
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={"steps": ss_sampling_steps, "cfg_strength": ss_guidance_strength},
            slat_sampler_params={"steps": slat_sampling_steps, "cfg_strength": slat_guidance_strength},
            mode=multiimage_algo,
        )

    video = render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return {"video_path": video_path, "state": state}
