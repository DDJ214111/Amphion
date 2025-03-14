# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from models.tts.metis.metis import Metis

from huggingface_hub import snapshot_download
from utils.util import load_config
import soundfile as sf

if __name__ == "__main__":

    device = "cuda:0"

    # TTS
    metis_cfg = load_config("./models/tts/metis/config/tts.json")

    ckpt_dir = snapshot_download(
        "amphion/maskgct",
        repo_type="model",
        local_dir="./models/tts/maskgct/ckpt",
        allow_patterns=["t2s/model.safetensors"],
    )

    ckpt_path = os.path.join(ckpt_dir, "t2s/model.safetensors")

    metis = Metis(
        ckpt_path=ckpt_path,
        cfg=metis_cfg,
        device=device,
        model_type="tts",
    )

    prompt_speech_path = "./models/tts/metis/wav/tts/prompt.wav"
    prompt_text = "코덱스 연금 급등 차트 탑쓰리. 연금투자, 신뢰가 중요하니까. 연금은 1등이 만든 삼성 코덱스ETF로"
    text = "코덱스 연금 급등 차트 탑쓰리. 연금투자, 신뢰가 중요하니까. 연금은 1등이 만든 삼성 코덱스ETF로"

    n_timesteps = 25
    cfg = 2.5
    model_type = "tts"

    gen_speech = metis(
        prompt_speech_path=prompt_speech_path,
        text=text,
        prompt_text=prompt_text,
        model_type=model_type,
        n_timesteps=n_timesteps,
        cfg=cfg,
    )

    sf.write("./models/tts/metis/wav/tts/gen.wav", gen_speech, 24000)
