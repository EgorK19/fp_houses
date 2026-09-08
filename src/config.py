from pathlib import Path

from omegaconf import OmegaConf

raw_data_dir = Path("d:/vs_projects/fp_houses/data/raw")

config_dict = {
    "general": {
        "seed": 101,
    },
    "paths": {
        "raw_dir": str(raw_data_dir),
        "train": str(raw_data_dir / "train.csv"),
        "test": str(raw_data_dir / "test.csv"),
    },
}


cfg = OmegaConf.create(config_dict)
