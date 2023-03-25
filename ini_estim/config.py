import json
import pathlib
from ini_estim import DIR


CONFIG_PATH = pathlib.Path(DIR, "config.json")

DEFAULT_CFG = {
    "dataset_directory": None,
}

def get_config():
    cfg = DEFAULT_CFG.copy()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    else:
        save_config(cfg)
    return cfg

def save_config(new_cfg):
    cfg = DEFAULT_CFG.copy()
    cfg.update(new_cfg)
    with open(CONFIG_PATH, 'w+') as f:
        json.dump(cfg, f)
