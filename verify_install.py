# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import math

# Define the file path to the configuration file.
config_file_path = 'config/config.json'

# Function to read the configuration file.
def read_config(file_path):
    try:
        # Open and read the file.
        with open(file_path, 'r', encoding='utf8') as file:
            return json.load(file)
    except FileNotFoundError:
        # Return None if the file is not found.
        return None

# Function to check if the engine file exists.
def check_engine_exists(model_path, engines):

    if engines != []:
        for engine in engines:
            # Construct the full path to the engine file.
                engine_path = os.path.join(os.getcwd(), model_path, engine)
                # Check if the file exists at the path.
                if not os.path.exists(engine_path):
                    return False
    else:
        engine_path = os.path.join(os.getcwd(), model_path)
        # Check if the file exists at the path.
        if not os.path.exists(engine_path):
            return False

    return True

def check_checkpoints_exists(checkpoints_path, checkpoints):
    if checkpoints != []:
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(os.getcwd() ,checkpoints_path, checkpoint)
            if not os.path.exists(checkpoint_path):
                return  False
        return True

    else:
        return False

# Function to save the configuration file.
def save_config(file_path, data):
    try:
        # Open and write the updated data to the file.
        with open(file_path, 'w', encoding='utf8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        # Print an error message if saving fails.
        print(f"Error saving the file: {e}")
        return False

# Read the configuration file.
config = read_config(config_file_path)
nvmlInit()
vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
total_vid_mem = math.ceil(vid_mem_info.total / (1024 * 1024 * 1024))
installed_model_found = False
if config:
    # Iterate through each model in the configuration.
    for model in config['models']['supported']:
        # Retrieve model path and engine file name.
        if "ngc_model_name" in model:
            model_engine_path = os.path.join(os.getcwd(), "model", model["id"], "engine")
            engine = [model['metadata']['engine']]
            model_checkpoints_path = os.path.join(os.getcwd(), "model", model["id"], model["prerequisite"]["checkpoints_local_dir"])
            checkpoint = model['prerequisite']['checkpoints_files']

        elif "hf_model_name" in model:
            model_engine_path = os.path.join(os.getcwd(), "model", model["id"])
            engine = []
            model_checkpoints_path = os.path.join(os.getcwd(), "model", model["id"])
            checkpoint = model['prerequisite']['checkpoints_files']

        # Update the 'installed' flag based on engine file existence.
        download_status = check_checkpoints_exists(model_checkpoints_path, checkpoint)
        engine_status = check_engine_exists(model_engine_path, engine)
        model["downloaded"] = download_status
        model["setup_finished"] = engine_status
        
        if model['setup_finished'] and not installed_model_found:
            config['models']['selected'] = model['name']
            installed_model_found = True

        # check if the model needs to show in the dropdown based in the memory requirements
        vid_mem_requried = model['min_gpu_memory']
        if vid_mem_requried <= total_vid_mem:
            model['should_show_in_UI'] = True
        else:
            model['should_show_in_UI'] = False

    # Check and update for ASR as well
    for model in config['models']['supported_asr']:
        engines=[]
        engines.append(model['metadata']['encoder_engine'])
        engines.append(model['metadata']['decoder_engine'])
        model_path = model['metadata']['model_path']
        # Update the 'installed' flag based on engine file existence.
        model['installed'] = check_engine_exists(model_path, engines)
        config["models"]["enable_asr"] = True

    # Save the updated configuration back to the file.
    if save_config(config_file_path, config):
        # Print confirmation and the updated configuration.
        print("App running with config\n", json.dumps(config, indent=4, ensure_ascii=False))
    else:
        # Print an error message if saving fails.
        print("Failed to save the updated configuration.")
