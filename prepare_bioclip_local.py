from __future__ import annotations

import os
from pathlib import Path

import open_clip


def main() -> None:
    project_root = Path(__file__).resolve().parent
    hf_home = Path(os.getenv('HF_HOME', project_root / 'models' / 'hf_cache')).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(hf_home)

    model_id = os.getenv('BIOCLIP_MODEL_ID', 'hf-hub:imageomics/bioclip')
    print(f'Preparing BioCLIP model: {model_id}')
    print(f'HF cache path: {hf_home}')

    open_clip.create_model_from_pretrained(model_id)
    print('BioCLIP model is cached locally and ready for offline usage.')


if __name__ == '__main__':
    main()
