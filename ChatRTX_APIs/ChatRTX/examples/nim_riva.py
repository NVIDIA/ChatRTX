
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

from openai import OpenAI
import os
from pathlib import Path
from ChatRTX.model_manager.nim_manager import NIMManager
from ChatRTX.inference.nims.riva.transcribe_file_offline import get_audio_to_text


def initialize_nim_manager(nim_profile):
    nim_manager = NIMManager.get_instance()
    if not nim_manager.download_nim(nim_profile):
        print(f"Could not download NIM {nim_profile}, exiting")
        exit()
    return nim_manager

def start_nim_service(nim_manager, nim_profile):
    http_ports = []
    grpc_ports = []
    if not nim_manager.start_nim(nim_profile, http_ports=http_ports, grpc_ports=grpc_ports, force_nim_start=True):
        print(f"Start NIM fail for {nim_profile}")
        exit()
    assert grpc_ports[0] == '50051' and len(grpc_ports)==1, "Example only provided for port 50051."

def main():
    
    # Update to path of audio file to be be transcribed.
    audio_path = "D:\\test.wav"

    print("Welcome to chat using ChatRTX APIs example!")

    nim_profile = "nvcr.io/nim/nvidia/parakeet-0-6b-ctc-en-us:2.0.0"


    try:
        nim_manager = initialize_nim_manager(nim_profile)
        start_nim_service(nim_manager, nim_profile)

        transcript = get_audio_to_text(
            server="localhost:50051",
            input_file=Path(audio_path),
            language_code="en-US",
            use_ssl=False
            )
        print("Transcript:", transcript)

    except Exception as e:
        print(f"An error occurred: {e}\n")
    finally:
        if 'nim_manager' in locals():
            nim_manager.stop_nim(nim_profile)

if __name__ == "__main__":
    main()

