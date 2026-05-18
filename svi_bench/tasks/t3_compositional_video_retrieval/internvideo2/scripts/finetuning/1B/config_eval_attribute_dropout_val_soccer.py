import os as __os
import pathlib as __pathlib


# Resolution: T3_ROOT env var, else <repo>/data/t3 (walking up to find
# pyproject.toml), else "T3_ROOT_NOT_SET" sentinel.
def __find_t3_root() -> str:
    if (env := __os.environ.get("T3_ROOT")):
        return env
    here = __pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            candidate = parent / "data" / "t3"
            if candidate.exists():
                return str(candidate)
            break
    return "T3_ROOT_NOT_SET"


__T3_ROOT = __pathlib.Path(__find_t3_root())
__T3_EMBEDS = __T3_ROOT / "embeds"

from configs.data import *
from configs.model import *
# ========================= data ==========================
# NOTE The train_file will not be used during the evaluation
train_file = available_corpus["sports_ret_300k_train_v2_concept"]  #sports_ret_300k_train
test_file = dict(
    soccer_val=available_corpus["soccer_val_v2"],
    # basketball_val=available_corpus["basketball_val_v2"],
    )

test_types = [
    "soccer_val",
    # "basketball_val"
    ]


num_workers = 6

best_key = ["soccer_val", "t2v_r1"]

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
        # backbone
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
        pretrained=str(__T3_ROOT / "ckpts" / "InternVideo2-stage2_1b-224p-f4.pt"),
        use_checkpoint=False,
        checkpoint_num=40,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        # clip teacher
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        # mask
        video_mask_type="random",
        video_mask_ratio=0.8,
        image_mask_type="random",
        image_mask_ratio=0.5,
        sep_image_video_pos_embed=True,
        keep_temporal=False,
        only_mask=True
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
    find_unused_parameters=True
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
        mlm=0.0, 
        vtm=0.0, 
        mvm=0.0,
        uta=0.0,
    ),  # 0: disabled.
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    distill_final_features=True,
    clip_loss_ratio=[1., 1.]
)

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.05,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # max_grad_norm=3.,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

# scheduler = dict(sched="cosine", epochs=1, min_lr_multi=0.01, warmup_epochs=0.2)
scheduler = dict(sched="cosine", epochs=5, min_lr_multi=0.01, warmup_epochs=0)

# zero_shot = True
evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=32,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
    embed_dir=str(__T3_EMBEDS / "embeds_val_soccer_partial.pt")
)

use_half_precision = True
use_bf16 = True

gradient_checkpointing = True # for text encoder
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

# ========================= wandb ==========================
# To enable wandb logging, set enable=True and fill in entity/project.
wandb = dict(
    enable=False,
    entity="",
    project="",
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42

save_latest = False
auto_resume = False
jump_evaluate = False
pretrained_path = ""

deepspeed = dict(
    enable=False,
    stage=1,
)
