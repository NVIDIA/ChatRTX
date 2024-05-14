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

import ngcsdk
import threading
import json
import os
import shutil
import builtins
import subprocess
import requests
from tqdm import tqdm
import time

# Capture the original print function
original_print = builtins.print

class OutputCapturePrint:
    def __init__(self):
        self.captured_output = []

    def custom_print(self, *args, **kwargs):
        """Custom print function to capture and print messages."""
        message = ' '.join(map(str, args)) + kwargs.get('end', '\n')
        # Use the original print to print to stdout
        original_print(message, end=kwargs.get('end', ''))
        # Capture the printed message
        self.captured_output.append(message)

    def get_captured_output(self):
        """Return the captured output."""
        return ''.join(self.captured_output)


def execute_command(command):
    """Executes a command in the command line."""
    try:
        # Launch the command and wait for it to finish
        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Decode and print the stdout and stderr from the command
        print(process.stdout.decode())
        print(process.stderr.decode())
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        print(e.stdout.decode())
        print(e.stderr.decode())


def build_engine_for_model(model_info, checkpoints_local_dir, engine_local_dir):
    # Read the command from model_info
    engine_build_cmd = model_info['prerequisite']['engine_build_command']
    # Replace placeholders with actual directory paths
    engine_build_cmd_formatted = engine_build_cmd.replace('%checkpoints_local_dir%', checkpoints_local_dir).replace(
        '%engine_dir%', engine_local_dir)

    # Execute the formatted command
    execute_command(engine_build_cmd_formatted)
    engine_path =  os.path.join(engine_local_dir, model_info['metadata']['engine'])
    if os.path.exists(engine_path):
        print("engine build")
        return True
    else:
        print("Fail to build the engine")
        return False


def download_model_with_monitoring(clt, model, path, output_capture):
    """Function to replace print, perform download, and restore print."""
    # Replace the built-in print with our custom_print
    builtins.print = output_capture.custom_print
    try:
        # Perform the download operation
        if model == "nvidia/llama/gemma-7b-int4-rtx:1.1":
            clt.registry.resource.download_version(model, path)
        else:
            clt.registry.model.download_version(model, path)
        
    finally:
        # Ensure the original print function is restored
        builtins.print = original_print

def parse_download_status(output):
    """Parse the download status and model path from captured output."""
    download_status = "Download Status: Failed"
    model_path = None
    for line in output.split('\n'):
        if "Download status:" in line:
            download_status = line.strip()
        if "Downloaded local path model:" in line:
            model_path = line.split(":", 1)[1].strip()
        if "Downloaded local path resource:" in line:
            model_path = line.split(":", 1)[1].strip()
    return "COMPLETED", model_path


def download_model(download_path, ngc_model_name):
    clt = ngcsdk.Client()
    output_capture = OutputCapturePrint()

    # Create a thread for the download process
    download_thread = threading.Thread(target=download_model_with_monitoring,
                                       args=(clt, ngc_model_name, download_path, output_capture))

    download_thread.start()
    download_thread.join()

    # After download, parse and print the status and path
    captured_output = output_capture.get_captured_output()
    download_status, model_path = parse_download_status(captured_output)
    if download_status == "COMPLETED":
        print(f"Download status: {download_status}")
        if model_path:
            print(f"Model Path: {model_path}")

    return download_status, model_path

def move_files(src_dir, dest_dir, file_names):
    """Move specified files from src_dir to dest_dir."""
    try:
        os.makedirs(dest_dir, exist_ok=True)  # Ensure destination directory exists
        for file_name in file_names:
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dest_dir, file_name)
            if os.path.isfile(src_path):
                shutil.move(src_path, dst_path)
                print(f"Moved: {file_name} to {dest_dir}")
            else:
                print(f"File not found: {src_path}")
                return False
        return True
    except Exception as e:
        print(f"An error occurred while moving files: {e}")
        return False

def process_model_files(model_info, download_dir, model_setup_path):
    try:
        # 1. Move checkpoint files
        checkpoints_files = model_info['prerequisite']['checkpoints_files']
        checkpoints_local_dir = os.path.join(model_setup_path, model_info['prerequisite']['checkpoints_local_dir'])
        if not move_files(download_dir, checkpoints_local_dir, checkpoints_files):
            print("Failed to move checkpoint files.")
            return False, "", ""

        # 2. Move tokenizer files
        tokenizer_files = list(model_info['prerequisite']['tokenizer_files'].values())
        if model_info['ngc_model_name'] == "nvidia/llama/gemma-7b-int4-rtx:1.1":
            tokenizer_local_dir = os.path.join(model_setup_path, model_info['prerequisite']['vocab_local_dir'])
        else:
            tokenizer_local_dir = os.path.join(model_setup_path, model_info['prerequisite']['tokenizer_local_dir'])
        tokenizer_ngc_dir = os.path.join(download_dir, model_info['prerequisite']['tokenizer_ngc_dir'])
        if not move_files(tokenizer_ngc_dir, tokenizer_local_dir, tokenizer_files):
            print("Failed to move tokenizer files.")
            return False, "", ""

        # 3. Ensure the engine directory exists
        engine_dir_path = os.path.join(model_setup_path, model_info['prerequisite']['engine_dir'])
        if not os.path.exists(engine_dir_path):
            os.makedirs(engine_dir_path)
        print(f"Directory created/exists: {engine_dir_path}")
        return True, checkpoints_local_dir, engine_dir_path
    except Exception as e:
        print(f"An error occurred during the process: {e}")
        return False

def download_file(url, destination):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(destination, 'wb') as file:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

def remove_directory(directory_path):
    try:
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Remove the directory and all its contents
            shutil.rmtree(directory_path)
            return True  # Successfully removed the directory
        else:
            return False  # Directory does not exist, no action needed
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def download_model_by_name(model_info, download_path):
    status = False
    if 'ngc_model_name' in model_info and model_info['ngc_model_name']:
        ngc_model_name = model_info['ngc_model_name']
        download_status, download_folder = download_model(download_path=download_path,
                                                          ngc_model_name=ngc_model_name)

        if download_status == "COMPLETED":
            model_setup_path = os.path.join(download_path, model_info['id'])
            status, checkpoints_local_dir, engine_local_dir = process_model_files(model_info, download_folder,
                                                                                  model_setup_path)
            if status:
                try:
                    shutil.rmtree(download_folder)
                    print(f"Successfully cleaned up download folder: {download_folder}")
                except Exception as e:
                    print(f"Failed to delete the folder: {e}")
                    status = False
            else:
                print(f"Failed to process model files for {model_info['name']}")
                status = False
        else:
            print(f"Download failed for model {model_info['name']} with status: {download_status}")
            status = False

        # Code to build the engine here
    elif 'hf_model_name' in model_info and model_info['hf_model_name']:
        model_setup_path = os.path.join(download_path, model_info['id'])
        os.makedirs(model_setup_path, exist_ok=True)
        checkpoints_files = model_info['prerequisite']['checkpoints_files']
        download_link = model_info['download_link']
        status = True
        for file in checkpoints_files:
            url = download_link + "/" + file + "?download=true"
            destination = os.path.join(model_setup_path, file)
            print(f"URL to download is {url}")
            try:
                download_file(url, destination)
                print(f"Download successful for the file {file}")
            except Exception as e:
                print(f"Download failed for the file {file}. Error: {e}")
                status = False

    if status == False:
        model_setup_path = os.path.join(download_path, model_info['id'])
        if os.path.exists(model_setup_path):
            remove_directory(model_setup_path)
    return status

def build_engine_by_name(model_info, download_path):
    try:
        status = True
        if model_info["is_installation_required"] == False:
            print("Engine build is not requied")
            time.sleep(2)
            return status
        if model_info["setup_finished"] == True:
            print("Engine is alread build")
            return status
        model_setup_path = os.path.join(download_path, model_info['id'])
        checkpoints_local_dir = os.path.join(model_setup_path, model_info['prerequisite']['checkpoints_local_dir'])
        engine_dir_path = os.path.join(model_setup_path, model_info['prerequisite']['engine_dir'])
        if not os.path.exists(engine_dir_path):
            os.makedirs(engine_dir_path)
        status = build_engine_for_model(model_info, checkpoints_local_dir, engine_dir_path)
        return status
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
