"""InternVideo2 eval-time config for T9 video-embedding extraction.

Loaded by ``demo/utils.py:setup_internvideo2``. Only the ``model``, ``inputs``,
``device``, ``use_half_precision`` / ``use_bf16``, and ``num_frames`` fields are
read at runtime; the rest is kept here (and matches the upstream eval-config
template) so the EasyDict-style config behaves the same as in upstream demos.

The actual sports-finetuned weights are loaded via ``configs/models.yaml``'s
``embedding_models.internvideo2.model_path`` (relative to T9_ROOT). The
``pretrained`` path under ``vision_encoder`` is a placeholder — the inner
weights it would otherwise load come from the SVI-Bench T3 base ckpt; if
unavailable, model init falls back gracefully.
"""

from configs.model import *

# ========================= input ==========================
num_frames = 16
num_frames_test = 16
batch_size = 2
batch_size_test = 4
max_txt_l = 200
origin_num_frames = 4

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
text_enc = "bert_large"
model = dict(
    model_cls="InternVideo2_Stage2",
    vision_encoder=dict(
        name="pretrain_internvideo2_1b_patch14_224",
        img_size=224,
        num_frames="${num_frames}",
        tubelet_size=1,
        patch_size=14,
        d_model=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        pretrained="",   # filled by run-time loader; see module docstring
        use_checkpoint=False,
        checkpoint_num=40,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        video_mask_type="random",
        video_mask_ratio=0.8,
        image_mask_type="random",
        image_mask_ratio=0.5,
        sep_image_video_pos_embed=True,
        keep_temporal=False,
        only_mask=True,
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
    find_unused_parameters=True,
)

evaluation = dict(
    eval_frame_ensemble="concat",
    eval_x_only=False,
    k_test=32,
    eval_offload=True,
)

use_half_precision = True
use_bf16 = True
gradient_checkpointing = True
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None
resume = False
debug = False
log_freq = 100
seed = 42
save_latest = False
auto_resume = False
jump_evaluate = True
pretrained_path = ""   # override at runtime via models.yaml -> model_path

deepspeed = dict(enable=False, stage=1)
