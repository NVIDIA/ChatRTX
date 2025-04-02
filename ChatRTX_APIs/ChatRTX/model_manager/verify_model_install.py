# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
import os
import math
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import subprocess
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set of GPUs that support NIMs (using trimmed IDs as integers)
gpu_ids_for_nims = {0x2684, 0x2704, 0x26B1, 0x2B85, 0x2C02}
gpu_ids_for_blackwell = {0x2B85, 0x2C02}
gpu_ids_for_ars_nims = {0x2684, 0x2704, 0x26B1, 0x2B85, 0x2C02}
gpu_ids_where_ars_diabled = {} # where no asr model supported (nothing for now)

def get_host_gpu_device_id():
    """
    Retrieves and trims the GPU device ID of the host using nvidia-smi by ignoring the last four characters.

    Returns:
        int or None: Trimmed GPU device ID as an integer, or None if retrieval fails.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=pci.device_id",
            "--format=csv,noheader"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        if output and len(output) == 10:
            trimmed_str = output[:-4]  # Trim the last four characters
            try:
                trimmed_id = int(trimmed_str, 16)
                return trimmed_id
            except ValueError:
                logger.error(f"Failed to convert trimmed GPU ID '{trimmed_str}' to integer.")
                return None
        else:
            logger.error("Unexpected output from nvidia-smi or output too short.")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi command failed: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving GPU device ID: {e}")
        return None


def check_nims_support():
    host_gpu_id = get_host_gpu_device_id()
    if host_gpu_id is None:
        return False
    if host_gpu_id in gpu_ids_for_nims:
        return True
    else:
        return False

def is_asr_supported():
    host_gpu_id = get_host_gpu_device_id()
    if host_gpu_id is None:
        return False

    if host_gpu_id in gpu_ids_where_ars_diabled:
        return False
    else:
        return True
    pass

def check_asr_nims_support():
    host_gpu_id = get_host_gpu_device_id()
    if host_gpu_id is None:
        return False

    if host_gpu_id in gpu_ids_for_ars_nims:
        return True
    else:
        return False

def is_riva_installed(config):
    try:
        # Get the local NIM list
        from ChatRTX.model_manager.nim_manager import NIMManager
        nim_manager = NIMManager.get_instance()
        local_nims = nim_manager.list_local_nims()

        image_name = None  # Initialize image_name to avoid reference before assignment

        # Read the Riva NIM image from config
        if 'models' in config and 'supported_asr' in config['models']:
            for model in config['models']['supported_asr']:
                if model.get("backend") == "nims":
                    image_name = model.get("image_name")
                    if image_name:
                        # Found the image name; exit the loop.
                        break
                    else:
                        logger.error(
                            f"image_name not found for ASR model: {model.get('name', 'unknown')}"
                        )

        if not image_name:
            logger.error("No valid Riva image name found in configuration.")
            return False

        # Check if the retrieved image is in the local NIM list
        return image_name in local_nims

    except Exception as e:
        logger.exception(f"Error in is_riva_installed: {e}")
        return False



def read_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return None

def check_engine_exists(model_path, engines):
    if engines:
        for engine in engines:
            engine_path = os.path.join(model_path, engine)
            if not os.path.exists(engine_path):
                return False
    else:
        if not os.path.exists(model_path):
            return False
    return True

def check_checkpoints_exists(checkpoints_path, checkpoints):
    if checkpoints:
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(checkpoints_path, checkpoint)
            if not os.path.exists(checkpoint_path):
                return False
        return True
    return False

def save_config(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving the file: {e}")
        return False

def update_config(models_dir, config_path):
    try:
        config = read_config(config_path)
        if not config:
            return

        nvmlInit()
        vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
        total_vid_mem = math.ceil(vid_mem_info.total / (1024 * 1024 * 1024))

        host_gpu_id = get_host_gpu_device_id()

        is_blackwell_gpu = True if host_gpu_id in gpu_ids_for_blackwell else False
        is_nim_supported = check_nims_support()

        if 'models' in config:
            for model in config['models']['supported']:
                # Ensure min_gpu_memory is an integer
                min_gpu_memory = model.get('min_gpu_memory', 0)
                if not isinstance(min_gpu_memory, int):
                    logger.error(f"Error: min_gpu_memory is not an integer for model {model.get('id', 'unknown')}.")
                    continue
                if min_gpu_memory > total_vid_mem:
                    model['should_show_in_UI'] = False
                elif (is_nim_supported and model['backend'] == "nims"):
                    model['should_show_in_UI'] = True
                elif is_nim_supported == False and model['backend'] == "nims":
                    model['should_show_in_UI'] = False
                elif (is_blackwell_gpu and (model['backend'] == "TRTLLM")):
                    model['should_show_in_UI'] = False
                else:
                    model['should_show_in_UI'] = True

            # Update model properties for 'supported' models
            for model in config['models']['supported']:
                try:
                    if model["backend"] == "TRTLLM":
                        model_engine_path = os.path.join(models_dir, model["id"], "engine")
                        model_checkpoints_path = os.path.join(models_dir, model["id"], model["prerequisite"]["checkpoints_local_dir"])

                        engine = [model['metadata']['engine']] if "ngc_model_name" in model else []
                        checkpoint = model['prerequisite']['checkpoints_files']

                        model["downloaded"] = check_checkpoints_exists(model_checkpoints_path, checkpoint)
                        model["setup_finished"] = check_engine_exists(model_engine_path, engine)
                except Exception as e:
                    logger.error(f"Error updating model {model.get('id', 'unknown')}: {e}")

            is_asr_enabled = False
            # Update models for ASR
            if 'supported_asr' in config['models']:
                for model in config['models']['supported_asr']:
                    if model["backend"] == "TRTLLM":
                        try:
                            engines = [model['metadata']['encoder_engine'], model['metadata']['decoder_engine']]
                            model_path = model['metadata']['model_path']
                            model['installed'] = (check_nims_support() == False) and is_asr_supported() and check_engine_exists(model_path, engines)
                            if model['installed'] == True:
                                is_asr_enabled = True
                        except Exception as e:
                            logger.error(f"Error updating ASR model {model.get('id', 'unknown')}: {e}")
                    elif model["backend"] == "nims":
                        model['installed'] = check_nims_support() and is_asr_supported() and is_riva_installed(config)
                        if model['installed'] == True:
                            is_asr_enabled = True
                    else:
                        logger.error(f"Error updating ASR model {model.get('id', 'unknown')}: {e}")

            config["models"]["enable_asr"] = is_asr_enabled
            # check if the installer have installed default model if yes, update the download and install key
            # Get default model from config
            default_model_id = config.get('models', {}).get('selected')
            if default_model_id:
                try:
                    # Import NIMManager to check local NIMs
                    from ChatRTX.model_manager.nim_manager import NIMManager
                    nim_manager = NIMManager.get_instance()
                    local_nims = nim_manager.list_local_nims()

                    # Find the default model in supported models
                    default_model = next((model for model in config['models']['supported']
                                       if model['id'] == default_model_id), None)
                    if default_model:
                        if default_model['backend'] == 'nims':
                            # Check if model's NIM ID exists in local NIMs
                            nim_id = default_model.get('nims_id')
                            if nim_id in local_nims:
                                default_model['downloaded'] = True
                                default_model['setup_finished'] = True
                            else:
                                default_model['downloaded'] = False
                                default_model['setup_finished'] = False
                except Exception as e:
                    logger.error(f"Error checking default model status: {e}")
            if save_config(config_path, config):
                logger.info("Config saved successfully.")
            else:
                logger.error("Failed to save the updated configuration.")

    except Exception as e:
        logger.critical(f"Critical error in update_config: {e}")

if __name__ == "__main__":
    update_config(
        models_dir="C:\\ProgramData\\NVIDIA Corporation\\ChatRTX\\models",
        config_path="C:\\ProgramData\\NVIDIA Corporation\\ChatRTX\\config\\config.json"
    )