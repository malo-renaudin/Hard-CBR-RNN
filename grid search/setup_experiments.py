"""Setup experiment configuration files for job array"""
import json
from pathlib import Path
from experiment_configs import get_all_configs


def create_configs():
    """Create all configuration files for job array"""
    configs = get_all_configs()
    
    # Create configs directory
    configs_dir = Path("job_cbr_2_configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Save each config
    for i, config in enumerate(configs):
        config_file = configs_dir / f"config_{i:03d}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Created {len(configs)} config files in {configs_dir}/")
    return len(configs)


if __name__ == "__main__":
    num_configs = create_configs()
    print(f"\nSetup complete!")
    print(f"Created {num_configs} configuration files")