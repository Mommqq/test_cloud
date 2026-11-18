"""Streamlit Cloud bootstrap.

The GitHub repository stays small. Public demo assets and the clean model are
downloaded from the already uploaded Hugging Face Space at first launch, then
cached by huggingface_hub on the Community Cloud volume.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

bundle = Path(
    snapshot_download(
        repo_id="heiheilxr/distributed_cloud_coding_test",
        repo_type="space",
        local_dir=Path.home() / ".cache" / "crdic_bundle",
        allow_patterns=[
            "assets/**",
            "checkpoints/crdic_clean.pth",
            "configs/train.json",
            "model/**",
            "crdic/**",
        ],
    )
)
os.environ["CRDIC_BUNDLE_ROOT"] = str(bundle)

from streamlit_app import *  # noqa: E402,F401,F403

