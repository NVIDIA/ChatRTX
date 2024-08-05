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

def read_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            return json.load(file)
    except FileNotFoundError:
        return None

def check_engine_exists(model_path, engines):
    if engines:
        for engine in engines:
            engine_path = os.path.join(model_path, engine)
            if not os.path.exists(engine_path):
                return False
    else:
        engine_path = model_path
        if not os.path.exists(engine_path):
            return False
    return True


def check_checkpoints_exists(checkpoints_path, checkpoints):
    if checkpoints:
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(checkpoints_path, checkpoint)
            if not os.path.exists(checkpoint_path):
                return False
        return True
    else:
        return False


def save_config(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving the file: {e}")
        return False


def update_config(models_dir, config_path):
    config = read_config(config_path)
    nvmlInit()
    vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    total_vid_mem = math.ceil(vid_mem_info.total / (1024 * 1024 * 1024))
    installed_model_found = False
    if config:
        for model in config['models']['supported']:
            if "ngc_model_name" in model:
                model_engine_path = os.path.join(models_dir, model["id"], "engine")
                engine = [model['metadata']['engine']]
                model_checkpoints_path = os.path.join(models_dir, model["id"],
                                                      model["prerequisite"]["checkpoints_local_dir"])
                checkpoint = model['prerequisite']['checkpoints_files']
            elif "hf_model_name" in model:
                model_engine_path = os.path.join(models_dir, model["id"])
                engine = []
                model_checkpoints_path = os.path.join(models_dir, model["id"])
                checkpoint = model['prerequisite']['checkpoints_files']

            download_status = check_checkpoints_exists(model_checkpoints_path, checkpoint)
            engine_status = check_engine_exists(model_engine_path, engine)
            model["downloaded"] = download_status
            model["setup_finished"] = engine_status

            if model['setup_finished'] and not installed_model_found:
                config['models']['selected'] = model['id']
                installed_model_found = True

            vid_mem_requried = model['min_gpu_memory']
            if vid_mem_requried <= total_vid_mem:
                model['should_show_in_UI'] = True
            else:
                model['should_show_in_UI'] = False

        for model in config['models']['supported_asr']:
            engines = [model['metadata']['encoder_engine'], model['metadata']['decoder_engine']]
            model_path = model['metadata']['model_path']
            model['installed'] = check_engine_exists(model_path, engines)
            config["models"]["enable_asr"] = True

        if save_config(config_path, config):
            print("Config saved")
        else:
            print("Failed to save the updated configuration.")

if __name__ == "__main__":
    update_config()
