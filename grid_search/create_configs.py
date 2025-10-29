from itertools import product
import json
from pathlib import Path

from itertools import product
import json
from pathlib import Path

from itertools import product
import json
from pathlib import Path


def create_configs(save_dir="./experiments/configs"):
    """Generate model-specific configuration files for each model type."""

    # ---- MODEL-SPECIFIC HYPERPARAMETERS ----
    grids = {
        "CBR_RNN": {
            "use_gumbel_softmax": [False],
            "end_temp": [1],
            "temp_decay": ['cosine'],
            "nhid": [1024],
            "nheads": [12],
            "nlayers":[1]
        },
        "Transformer": {
            "use_gumbel_softmax": [False],
            "end_temp": [1],
            "temp_decay": ['cosine'],
            "nhid": [1024],
            "nheads": [12],
            "nlayers": [12],
        },
        # "LSTM": {
        #     "nhid": [256, 512, 1024],
        # },
    }

    # ---- GENERATE CONFIGURATIONS ----
    configs = []
    for model_name, param_grid in grids.items():
        keys, vals = zip(*param_grid.items())

        for combo in product(*vals):
            cfg = dict(zip(keys, combo))
            cfg["model"] = model_name

            # ensure compatibility (e.g., nhid divisible by nheads)
            if "nheads" in cfg and cfg["nhid"] % cfg["nheads"] != 0:
                continue

            configs.append(cfg)

    # ---- SAVE TO DISK ----
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, config in enumerate(configs):
        with open(save_dir / f"config_{i:03d}.json", "w") as f:
            json.dump(config, f, indent=2)

    print(f"✅ Generated {len(configs)} configurations in {save_dir}")
    return configs


if __name__ == "__main__":
    create_configs()
