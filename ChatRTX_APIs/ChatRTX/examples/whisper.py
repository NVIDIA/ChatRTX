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

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from ChatRTX.inference.trtllm.whisper.trt_whisper import WhisperTRTLLM, decode_audio_file
from ChatRTX.inference.trtllm.whisper.whisper_utils import process_input_audio
import time

class WhisperASR:
    def __init__(self, asr_engine_path, asr_assets_path, audio_path):
        self.asr_engine_path = asr_engine_path
        self.asr_assets_path = asr_assets_path
        self.audio_path = audio_path
        self.whisper_model = None
        self.whisper_model_loaded = False
        self.enable_asr = False
        nvmlInit()

    def mic_init_handler(self):
        self.enable_asr = True
        if not self.enable_asr:
            return False
        vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
        free_vid_mem = vid_mem_info.free / (1024 * 1024)
        print("free video memory in MB = ", free_vid_mem)
        if self.whisper_model is not None:
            self.whisper_model.unload_model()
            del self.whisper_model
            self.whisper_model = None
        self.whisper_model = WhisperTRTLLM(self.asr_engine_path, assets_dir=self.asr_assets_path)
        self.whisper_model_loaded = True
        return

    def mic_recording_done_handler(self):
        transcription = ""
        if not self.enable_asr:
            return ""
        # Check and wait until model is loaded before running it.
        checks_for_model_loading = 40
        checks_left_for_model_loading = checks_for_model_loading
        sleep_time = 0.2
        while checks_left_for_model_loading > 0 and not self.whisper_model_loaded:
            time.sleep(sleep_time)
            checks_left_for_model_loading -= 1
        if checks_left_for_model_loading == 0:
            return ""
        new_file_path = process_input_audio(self.audio_path)
        language = "english"
        transcription = decode_audio_file(new_file_path, self.whisper_model, language=language, mel_filters_dir=self.asr_assets_path)
        if self.whisper_model is not None:
            self.whisper_model.unload_model()
            del self.whisper_model
            self.whisper_model = None
            self.whisper_model_loaded = False
        return transcription

# For instructions on getting Whisper Assets, generation engine refer:
# https://github.com/NVIDIA/ChatRTX/tree/release/0.3?tab=readme-ov-file#setup-whisper-medium-int8-model
asr_engine_path = "C:\\ProgramData\\NVIDIA Corporation\\ChatRTX\\models\\whisper\\whisper_medium_int8_engine"
asr_assets_path = "C:\\ProgramData\\NVIDIA Corporation\\ChatRTX\\models\\whisper\\whisper_assets"

# Path to audio file to be trascribed
audio_path = "C:\\test.wav"

whisper_asr = WhisperASR(asr_engine_path, asr_assets_path, audio_path)
whisper_asr.mic_init_handler()
result = whisper_asr.mic_recording_done_handler()
print(result)
