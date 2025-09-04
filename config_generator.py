#!/usr/bin/env python3
"""
Interactive configuration generator for language model training
Creates configs for grid search and launches job arrays
"""

import os
import yaml
import itertools
from pathlib import Path
from typing import Dict, Any, List
import argparse


def ask_question(question: str, options: List[str] = None, default: str = None, data_type: type = str):
    """Ask a question with optional validation"""
    while True:
        if options:
            print(f"\n{question}")
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
            if default:
                prompt = f"Choose (1-{len(options)}) [default: {default}]: "
            else:
                prompt = f"Choose (1-{len(options)}): "
        else:
            if default:
                prompt = f"{question} [default: {default}]: "
            else:
                prompt = f"{question}: "
        
        response = input(prompt).strip()
        
        if not response and default:
            return default if not options else options[int(default) - 1]
        
        if options:
            try:
                choice = int(response)
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
                    continue
            except ValueError:
                print("Please enter a valid number")
                continue
        else:
            try:
                if data_type == bool:
                    return response.lower() in ['true', 't', 'yes', 'y', '1']
                elif data_type == int:
                    return int(response)
                elif data_type == float:
                    return float(response)
                else:
                    return response
            except ValueError:
                print(f"Please enter a valid {data_type.__name__}")
                continue


def ask_list_values(question: str, data_type: type = str, min_values: int = 1):
    """Ask for multiple values for grid search"""
    print(f"\n{question}")
    print("Enter values separated by commas (e.g., 0.001,0.01,0.1)")
    
    while True:
        response = input("Values: ").strip()
        if not response:
            print("Please enter at least one value")
            continue
            
        try:
            values = [data_type(v.strip()) for v in response.split(',')]
            if len(values) < min_values:
                print(f"Please enter at least {min_values} value(s)")
                continue
            return values
        except ValueError:
            print(f"Please enter valid {data_type.__name__} values")
            continue


def configure_basic_settings():
    """Configure basic training settings"""
    print("\n" + "="*50)
    print("BASIC TRAINING CONFIGURATION")
    print("="*50)
    
    config = {}
    
    # Dataset configuration
    print("\n--- Dataset Configuration ---")
    config['dataset_path'] = ask_question(
        "Dataset path", 
        default="cbr_lightning/wikitext-103-tokenized"
    )
    config['tokenizer_path'] = ask_question(
        "Tokenizer path", 
        default="./tokenizer.pkl"
    )
    
    # Training basics
    print("\n--- Training Configuration ---")
    config['max_epochs'] = ask_question(
        "Maximum epochs", 
        default="50", 
        data_type=int
    )
    config['batch_size'] = ask_question(
        "Batch size", 
        default="32", 
        data_type=int
    )
    config['max_length'] = ask_question(
        "Maximum sequence length", 
        default="128", 
        data_type=int
    )
    
    # Hardware configuration
    print("\n--- Hardware Configuration ---")
    config['precision'] = ask_question(
        "Training precision",
        options=["32", "16-mixed", "bf16-mixed"],
        default="2"
    )
    config['num_workers'] = ask_question(
        "Number of data workers", 
        default="4", 
        data_type=int
    )
    
    return config


def configure_model_params(model_type: str):
    """Configure model-specific parameters"""
    print(f"\n--- {model_type} Model Configuration ---")
    
    if model_type == "CBR_RNN":
        return configure_cbr_rnn()
    elif model_type == "Transformer":
        return configure_transformer()
    elif model_type == "LSTM":
        return configure_lstm()


def configure_cbr_rnn():
    """Configure CBR_RNN specific parameters"""
    config = {}
    
    # Architecture parameters
    config['ninp'] = ask_question("Embedding dimension", default="256", data_type=int)
    config['nhid'] = ask_question("Hidden dimension", default="512", data_type=int)
    config['nheads'] = ask_question("Number of attention heads", default="4", data_type=int)
    config['compressed_dim'] = ask_question("Compressed dimension", default="32", data_type=int)
    config['dropout'] = ask_question("Dropout rate", default="0.1", data_type=float)
    
    # CBR-specific parameters
    config['temperature'] = ask_question("Temperature", default="1.0", data_type=float)
    config['gumbel_softmax'] = ask_question("Use Gumbel softmax", default="false", data_type=bool)
    
    if config['gumbel_softmax']:
        config['temp_decay_rate'] = ask_question("Temperature decay rate", default="0.95", data_type=float)
        config['temp_final'] = ask_question("Final temperature", default="0.1", data_type=float)
    
    return config


def configure_transformer():
    """Configure Transformer specific parameters"""
    config = {}
    
    config['d_model'] = ask_question("Model dimension (d_model)", default="384", data_type=int)
    config['n_heads'] = ask_question("Number of attention heads", default="8", data_type=int)
    config['n_layers'] = ask_question("Number of layers", default="6", data_type=int)
    config['d_ff'] = ask_question("Feed-forward dimension", default="1536", data_type=int)
    config['dropout'] = ask_question("Dropout rate", default="0.1", data_type=float)
    
    # Transformer-specific parameters
    config['temperature'] = ask_question("Temperature", default="1.0", data_type=float)
    config['gumbel_softmax'] = ask_question("Use Gumbel softmax", default="false", data_type=bool)
    
    return config


def configure_lstm():
    """Configure LSTM specific parameters"""
    config = {}
    
    config['embedding_dim'] = ask_question("Embedding dimension", default="256", data_type=int)
    config['hidden_dim'] = ask_question("Hidden dimension", default="512", data_type=int)
    
    return config


def configure_grid_search():
    """Configure grid search parameters"""
    print("\n" + "="*50)
    print("GRID SEARCH CONFIGURATION")
    print("="*50)
    
    enable_grid_search = ask_question(
        "Enable grid search?", 
        options=["yes", "no"], 
        default="1"
    ) == "yes"
    
    if not enable_grid_search:
        return {}
    
    print("\nSelect parameters to grid search over:")
    print("You can specify multiple values for each parameter")
    
    grid_params = {}
    
    # Learning rate
    if ask_question("Grid search learning rate?", options=["yes", "no"], default="1") == "yes":
        grid_params['learning_rate'] = ask_list_values("Learning rates", float)
    
    # Weight decay
    if ask_question("Grid search weight decay?", options=["yes", "no"], default="2") == "yes":
        grid_params['weight_decay'] = ask_list_values("Weight decay values", float)
    
    # Gradient clipping
    if ask_question("Grid search gradient clipping?", options=["yes", "no"], default="2") == "yes":
        grid_params['gradient_clip_val'] = ask_list_values("Gradient clip values", float)
    
    # Batch size
    if ask_question("Grid search batch size?", options=["yes", "no"], default="2") == "yes":
        grid_params['batch_size'] = ask_list_values("Batch sizes", int)
    
    # Dropout (if applicable)
    if ask_question("Grid search dropout?", options=["yes", "no"], default="2") == "yes":
        grid_params['dropout'] = ask_list_values("Dropout rates", float)
    
    return grid_params


def generate_config_combinations(base_config: Dict, grid_params: Dict) -> List[Dict]:
    """Generate all combinations of grid search parameters"""
    if not grid_params:
        return [base_config]
    
    # Get all parameter names and their values
    param_names = list(grid_params.keys())
    param_values = [grid_params[name] for name in param_names]
    
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    configs = []
    for i, combo in enumerate(combinations):
        config = base_config.copy()
        
        # Update config with current combination
        for param_name, param_value in zip(param_names, combo):
            if param_name in ['batch_size', 'gradient_clip_val']:
                # These go in trainer config
                config['trainer'][param_name] = param_value
            elif param_name in ['learning_rate', 'weight_decay', 'dropout']:
                # These go in model config
                config['model']['config'][param_name] = param_value
        
        # Add unique identifier
        config['experiment_name'] = f"{config['model']['type'].lower()}_run_{i:03d}"
        
        configs.append(config)
    
    return configs


def create_base_config(basic_settings: Dict, model_type: str, model_config: Dict) -> Dict:
    """Create base configuration structure"""
    
    config = {
        'seed': 42,
        'model': {
            'type': model_type,
            'config': {
                'learning_rate': 1e-3,
                'weight_decay': 0.01,
                **model_config
            }
        },
        'data': {
            'dataset_path': basic_settings['dataset_path'],
            'tokenizer_path': basic_settings['tokenizer_path'],
            'batch_size': basic_settings['batch_size'],
            'num_workers': basic_settings['num_workers'],
            'max_length': basic_settings['max_length']
        },
        'trainer': {
            'max_epochs': basic_settings['max_epochs'],
            'accelerator': 'auto',
            'devices': 'auto',
            'precision': basic_settings['precision'],
            'gradient_clip_val': 0.5,
            'accumulate_grad_batches': 1,
            'val_check_interval': 0.5
        },
        'logging': {
            'experiment_name': f'{model_type.lower()}_experiments',
            'tracking_uri': './mlruns'
        },
        'checkpoint': {
            'dirpath': f'./checkpoints/{model_type.lower()}',
            'monitor': 'val_loss',
            'mode': 'min',
            'save_top_k': 3,
            'save_every_epoch': True
        },
        'early_stopping': {
            'enabled': True,
            'monitor': 'val_loss',
            'patience': 10,
            'mode': 'min'
        },
        'run_test': True,
        'save_path': f'./final_models/{model_type.lower()}_final.ckpt'
    }
    
    return config


def save_configs_and_create_job_script(configs: List[Dict], output_dir: str = "configs"):
    """Save all configs and create robust job submission system with timeout handling"""
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "status").mkdir(exist_ok=True)
    
    config_files = []
    
    # Save each config
    for i, config in enumerate(configs):
        filename = f"{config['experiment_name']}.yaml"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        config_files.append(str(filepath))
        print(f"Config saved: {filepath}")
    
    # Create main job script with timeout handling
    job_script = f"""#!/bin/bash
#SBATCH --job-name=lm_training
#SBATCH --array=0-{len(configs)-1}
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90

# Timeout handler - save checkpoint before timeout
timeout_handler() {{
    echo "Timeout signal received, attempting graceful shutdown..."
    # Send SIGTERM to training process if it exists
    if [[ ! -z "$TRAINING_PID" ]]; then
        kill -TERM $TRAINING_PID
        wait $TRAINING_PID
    fi
    # Mark job as timed out for relaunch
    echo "TIMEOUT" > "status/job_${{SLURM_ARRAY_TASK_ID}}.status"
    exit 124  # Standard timeout exit code
}}

trap timeout_handler SIGUSR1

# Create necessary directories
mkdir -p logs status checkpoints

# Array of config files
CONFIGS=({' '.join([f'"{cf}"' for cf in config_files])})

# Get current config
CONFIG=${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}
JOB_ID="${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
STATUS_FILE="status/job_${{SLURM_ARRAY_TASK_ID}}.status"

echo "Starting job $SLURM_ARRAY_TASK_ID with config: $CONFIG"
echo "STARTED" > "$STATUS_FILE"

# Activate conda environment
source ~/.bashrc
conda activate leaps3

# Function to find latest comprehensive checkpoint
find_latest_checkpoint() {{
    local config_name=$(basename "$CONFIG" .yaml)
    local checkpoint_dir="checkpoints/${{config_name}}"
    if [[ -d "$checkpoint_dir" ]]; then
        local latest=$(ls -1 "$checkpoint_dir"/comprehensive_epoch_*.ckpt 2>/dev/null | sort -V | tail -1)
        echo "$latest"
    fi
}}

# Check for existing checkpoint to resume from
LATEST_CHECKPOINT=$(find_latest_checkpoint)

if [[ -n "$LATEST_CHECKPOINT" && -f "$LATEST_CHECKPOINT" ]]; then
    echo "Found existing checkpoint: $LATEST_CHECKPOINT"
    echo "Resuming training from checkpoint..."
    python cbr_lightning/train2.py --config "$CONFIG" --resume-comprehensive "$LATEST_CHECKPOINT" &
else
    echo "Starting fresh training..."
    python cbr_lightning/train2.py --config "$CONFIG" &
fi

TRAINING_PID=$!

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Record final status
if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo "COMPLETED" > "$STATUS_FILE"
    echo "Job $SLURM_ARRAY_TASK_ID completed successfully"
elif [[ $TRAINING_EXIT_CODE -eq 124 ]]; then
    echo "TIMEOUT" > "$STATUS_FILE"
    echo "Job $SLURM_ARRAY_TASK_ID timed out"
else
    echo "FAILED" > "$STATUS_FILE"
    echo "Job $SLURM_ARRAY_TASK_ID failed with exit code $TRAINING_EXIT_CODE"
fi

exit $TRAINING_EXIT_CODE
"""
    
    # Save main job script
    job_script_path = output_path / "submit_jobs.sh"
    with open(job_script_path, 'w') as f:
        f.write(job_script)
    os.chmod(job_script_path, 0o755)
    
    # Create job monitoring and relaunch script
    monitor_script = f"""#!/bin/bash
# Job monitoring and relaunch script
# Usage: ./monitor_and_relaunch.sh [job_id]

CONFIGS_DIR="{output_dir}"
cd "$CONFIGS_DIR"

# Function to check job status
check_job_status() {{
    local job_id="$1"
    if [[ -z "$job_id" ]]; then
        echo "Usage: check_job_status <job_id>"
        return 1
    fi
    
    echo "Checking status of job array $job_id..."
    
    local total_jobs={len(configs)}
    local completed=0
    local failed=0
    local timeout=0
    local running=0
    
    for i in $(seq 0 $((total_jobs-1))); do
        local status_file="status/job_${{i}}.status"
        if [[ -f "$status_file" ]]; then
            local status=$(cat "$status_file")
            case "$status" in
                "COMPLETED") ((completed++)) ;;
                "FAILED") ((failed++)) ;;
                "TIMEOUT") ((timeout++)) ;;
                "STARTED") ((running++)) ;;
            esac
        else
            ((running++))  # Assume still running if no status file
        fi
    done
    
    echo "Job Status Summary:"
    echo "  Completed: $completed"
    echo "  Running: $running" 
    echo "  Timed out: $timeout"
    echo "  Failed: $failed"
    echo "  Total: $total_jobs"
    
    return 0
}}

# Function to relaunch timed out or failed jobs
relaunch_failed_jobs() {{
    local job_id="$1"
    local total_jobs={len(configs)}
    local jobs_to_relaunch=()
    
    echo "Checking for jobs to relaunch..."
    
    for i in $(seq 0 $((total_jobs-1))); do
        local status_file="status/job_${{i}}.status"
        if [[ -f "$status_file" ]]; then
            local status=$(cat "$status_file")
            if [[ "$status" == "TIMEOUT" || "$status" == "FAILED" ]]; then
                jobs_to_relaunch+=($i)
                echo "Job $i marked for relaunch (status: $status)"
            fi
        fi
    done
    
    if [[ ${{#jobs_to_relaunch[@]}} -eq 0 ]]; then
        echo "No jobs need relaunching"
        return 0
    fi
    
    # Create relaunch array
    local relaunch_array=$(IFS=,; echo "${{jobs_to_relaunch[*]}}")
    echo "Relaunching jobs: $relaunch_array"
    
    # Submit relaunch job
    sbatch --array="$relaunch_array" submit_jobs.sh
    
    echo "Relaunch job submitted"
}}

# Function to wait for all jobs to complete and auto-relaunch
monitor_and_auto_relaunch() {{
    local job_id="$1"
    local max_retries=3
    local retry_count=0
    
    while [[ $retry_count -lt $max_retries ]]; do
        echo "\\n=== Monitoring cycle $((retry_count + 1)) ==="
        
        # Wait for current jobs to finish
        if [[ -n "$job_id" ]]; then
            echo "Waiting for job array $job_id to complete..."
            while squeue -j "$job_id" &>/dev/null; do
                sleep 60
            done
        fi
        
        # Check status
        check_job_status "$job_id"
        
        # Count jobs that need relaunching
        local total_jobs={len(configs)}
        local jobs_to_relaunch=0
        
        for i in $(seq 0 $((total_jobs-1))); do
            local status_file="status/job_${{i}}.status"
            if [[ -f "$status_file" ]]; then
                local status=$(cat "$status_file")
                if [[ "$status" == "TIMEOUT" || "$status" == "FAILED" ]]; then
                    ((jobs_to_relaunch++))
                fi
            fi
        done
        
        if [[ $jobs_to_relaunch -eq 0 ]]; then
            echo "All jobs completed successfully!"
            break
        fi
        
        echo "Found $jobs_to_relaunch jobs to relaunch"
        
        if [[ $retry_count -lt $((max_retries - 1)) ]]; then
            relaunch_failed_jobs "$job_id"
            ((retry_count++))
            
            # Get new job ID from the relaunch
            sleep 5
            job_id=$(squeue -u $USER --name=lm_training -h -o "%A" | tail -1)
            echo "New job ID: $job_id"
        else
            echo "Maximum retries ($max_retries) reached. Some jobs may require manual intervention."
            break
        fi
    done
}}

# Main script logic
case "${{1:-}}" in
    "status")
        check_job_status "$2"
        ;;
    "relaunch")
        relaunch_failed_jobs "$2"
        ;;
    "monitor")
        monitor_and_auto_relaunch "$2"
        ;;
    *)
        echo "Job Monitor and Relaunch System"
        echo "Usage:"
        echo "  ./monitor_and_relaunch.sh status [job_id]    - Check job status"
        echo "  ./monitor_and_relaunch.sh relaunch [job_id]  - Relaunch failed/timeout jobs"
        echo "  ./monitor_and_relaunch.sh monitor [job_id]   - Auto-monitor and relaunch"
        echo ""
        echo "Example workflow:"
        echo "  1. sbatch submit_jobs.sh                     # Submit initial jobs"
        echo "  2. ./monitor_and_relaunch.sh monitor 12345   # Auto-monitor and relaunch"
        ;;
esac
"""
    
    monitor_script_path = output_path / "monitor_and_relaunch.sh"
    with open(monitor_script_path, 'w') as f:
        f.write(monitor_script)
    os.chmod(monitor_script_path, 0o755)
    
    # Create simple submission wrapper
    submit_wrapper = f"""#!/bin/bash
# Simple submission wrapper with automatic monitoring
echo "Submitting {len(configs)} training jobs..."
JOB_ID=$(sbatch submit_jobs.sh | grep -o '[0-9]*')

if [[ -n "$JOB_ID" ]]; then
    echo "Job array submitted with ID: $JOB_ID"
    echo "Starting automatic monitoring..."
    ./monitor_and_relaunch.sh monitor "$JOB_ID"
else
    echo "Failed to submit jobs"
    exit 1
fi
"""
    
    submit_wrapper_path = output_path / "submit_and_monitor.sh"
    with open(submit_wrapper_path, 'w') as f:
        f.write(submit_wrapper)
    os.chmod(submit_wrapper_path, 0o755)
    
    print(f"\nRobust job system created in: {output_path}")
    print(f"Files created:")
    print(f"  - submit_jobs.sh (main job script)")
    print(f"  - monitor_and_relaunch.sh (monitoring system)")  
    print(f"  - submit_and_monitor.sh (one-click solution)")
    print(f"\nUsage options:")
    print(f"  Simple: cd {output_dir} && ./submit_and_monitor.sh")
    print(f"  Manual: cd {output_dir} && sbatch submit_jobs.sh")
    print(f"  Monitor: cd {output_dir} && ./monitor_and_relaunch.sh monitor <job_id>")
    
    return config_files, job_script_path


def main():
    parser = argparse.ArgumentParser(description='Interactive config generator for language model training')
    parser.add_argument('--output-dir', default='configs', help='Output directory for configs')
    parser.add_argument('--model-type', choices=['CBR_RNN', 'Transformer', 'LSTM'], help='Model type (skip interactive selection)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LANGUAGE MODEL TRAINING - CONFIG GENERATOR")
    print("="*60)
    
    # Select model type
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = ask_question(
            "Select model type",
            options=["CBR_RNN", "Transformer", "LSTM"]
        )
    
    print(f"\nConfiguring {model_type} model...")
    
    # Configure basic settings
    basic_settings = configure_basic_settings()
    
    # Configure model-specific parameters
    model_config = configure_model_params(model_type)
    
    # Configure grid search
    grid_params = configure_grid_search()
    
    # Create base configuration
    base_config = create_base_config(basic_settings, model_type, model_config)
    
    # Generate all configuration combinations
    all_configs = generate_config_combinations(base_config, grid_params)
    
    print(f"\nGenerated {len(all_configs)} configuration(s)")
    
    # Save configs and create job script
    config_files, job_script = save_configs_and_create_job_script(all_configs, args.output_dir)
    
    print("\nConfiguration generation complete!")
    print(f"Configs saved in: {args.output_dir}/")
    print(f"Job script: {job_script}")


if __name__ == '__main__':
    main()