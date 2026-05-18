import os as __os  # add "__" if not want to be exported
import pathlib as __pathlib
from copy import deepcopy as __deepcopy


# T3_ROOT is the directory containing data/, embeds/, ckpts/, compositions/
# for the SVI-Bench T3 task. Inside the SVI-Bench package (here:
# svi_bench/tasks/t3_compositional_video_retrieval/internvideo2/configs/) there is no fixed relative
# path to that data, so users must set T3_ROOT explicitly via env var when
# invoking the InternVideo2 finetune/eval shell scripts.
#
# T3 evaluation via `svi-bench evaluate --task t3 ...` does NOT load this file
# (it uses the lightweight retrieval.py path with cached embeddings).
#
# Resolution order:
#   1. T3_ROOT env var.
#   2. <repo>/data/t3, where <repo> is the SVI-Bench root (walking up from
#      this file to find pyproject.toml).
#   3. "T3_ROOT_NOT_SET" sentinel — module imports cleanly; failure deferred
#      to file-open time.
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


__T3_ROOT_STR = __find_t3_root()


def __resolve_t3_path(*parts: str) -> str:
    return str(__pathlib.Path(__T3_ROOT_STR, *parts))


# ============== pretraining datasets=================
available_corpus = dict(
    # pretraining image datasets
    cc3m=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image"
    ),
    cc12m=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image"
    ),
    sbu=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image"
    ),
    vg=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    coco=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    laion_2b=dict(
        anno_path="your_path",
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    laion_coco=dict(
        anno_path="your_path",
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    laion_pop=dict(
        anno_path="your_path",
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    # pretraining video datasets
    webvid_fuse_10m=dict(
        anno_path="your_path", 
        data_root="",
        media_type="video",
        jump_filter=True
    ),
    internvid_v1=dict(
        anno_path="your_path",
        data_root="",
        media_type="video",
        jump_filter=True
    ),
    internvid_v2_avs_private=dict( 
        anno_path="your_path",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        zero_audio_padding_for_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all'),
        jump_filter=True
    ),
    webvid=dict(
        anno_path="your_path",
        data_root="",
        media_type="video"
    ),
    webvid_10m=dict(
        anno_path="your_path",
        data_root="",
        media_type="video",
    ),
    # audio-text
    wavcaps_400k=dict(
        anno_path="your_path",
        data_root="",
        media_type="audio"
    ),
    # debug
    cc3m_debug=dict(
        anno_path="your_path",
        data_root="",
        media_type="image"
    ),
    webvid_debug=dict(
        anno_path="your_path",
        data_root="",
        media_type="video"
    )
)

available_corpus["pretrain_example_data_1B"] = [
    available_corpus['cc3m'], 
    available_corpus['webvid']
]

available_corpus["pretrain_example_data_6B"] = [
    available_corpus['cc3m'], 
    available_corpus['webvid'], 
    available_corpus['internvid_v2_avs_private']
]

available_corpus["data_25m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]

available_corpus["debug"] = [
    available_corpus["cc3m_debug"],
    available_corpus["webvid_debug"],
]

# ============== SVI-Bench T3 (sports retrieval) =================
# Sports retrieval annotation paths. Override via T3_ROOT env var.

# Each entry's `video` field is a relative path of the form
# "{sport}/{video_id}.mp4"; the loader joins it with data_root (= clips/).
__T3_CLIPS = __resolve_t3_path("clips")

# Train sets — "single" is one paraphrase per clip; "multi" is multiple paraphrases
# per clip, used by the attribute-dropout ("concept") training regime.
available_corpus["sports_ret_300k_train_v2"] = dict(
    anno_path=__resolve_t3_path("data", "train", "train_single.json"),
    data_root=__T3_CLIPS,
    media_type="video",
    max_txt_l=200,
)

available_corpus["sports_ret_300k_train_v2_concept"] = dict(
    anno_path=__resolve_t3_path("data", "train", "train_multi.json"),
    data_root=__T3_CLIPS,
    media_type="video",
    max_txt_l=200,
)

# Eval (val + test): one positive + 5,000 same-sport negatives per query.
for __sport in ("basketball", "hockey", "soccer"):
    for __split in ("val", "test"):
        available_corpus[f"{__sport}_{__split}_v2"] = dict(
            anno_path=__resolve_t3_path("data", __split, f"{__sport}_{__split}.json"),
            data_root=__T3_CLIPS,
            media_type="video",
            max_txt_l=200,
        )
del __sport, __split

# msrvtt_1k_test omitted in T3 release (not used).

available_corpus["didemo_ret_test"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["anet_ret_val"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["lsmdc_ret_test_1000"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video"
)

available_corpus["vatex_ch_ret_val"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video"
)

available_corpus["vatex_en_ret_val"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video"
)

available_corpus["k400_act_val"] = dict(
    anno_path="your_path",
    data_root="",
    is_act_rec=True,
)

available_corpus["k600_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["k700_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["mit_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["ucf101_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["hmdb51_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["ssv2_mc_val"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
)

available_corpus["charades_mc_test"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
)


available_corpus["anet_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["didemo_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64 
)

available_corpus["didemo_ret_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["lsmdc_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=96
)

available_corpus["lsmdc_ret_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=96
)

available_corpus["msrvtt_ret_train9k"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
)

available_corpus["msrvtt_ret_test1k"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
)

available_corpus["msvd_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=64,
    has_multi_txt_gt=True
)

available_corpus["msvd_ret_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=64
)

available_corpus["msvd_ret_test"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=64
)


available_corpus["vatex_en_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    has_multi_txt_gt=True
)


# audio-text

available_corpus["audiocaps_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)

available_corpus["audiocaps_ret_test"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)


available_corpus["clothov1_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)

available_corpus["clothov1_ret_test"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)

available_corpus["clothov2_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)

available_corpus["clothov2_ret_test"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)