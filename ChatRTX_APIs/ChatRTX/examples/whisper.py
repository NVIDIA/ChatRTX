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


asr_engine_path = "C:\\neva-git\\trt-llm-rag-windows\\ChatRTX\\model\\whisper\\whisper_medium_int8_engine"
asr_assets_path = "C:\\neva-git\\trt-llm-rag-windows\\ChatRTX\\model\\whisper\\whisper_assets"
selected_ChatGLM= False
whisper_model_loaded = False
whisper_model = None
enable_asr = False
audio_path="C:\\neva-git\\todelete\\1221-135766-0002.wav"


def mic_init_handler():
    global whisper_model, whisper_model_loaded, enable_asr
    enable_asr = True
    if not enable_asr:
        return False
    vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    free_vid_mem = vid_mem_info.free / (1024*1024)
    print("free video memory in MB = ", free_vid_mem)
    if whisper_model is not None:
        whisper_model.unload_model()
        del whisper_model
        whisper_model = None
    whisper_model = WhisperTRTLLM(asr_engine_path, assets_dir=asr_assets_path)
    whisper_model_loaded = True
    return True


def mic_recording_done_handler(audio_path):
    transcription = ""
    global whisper_model, enable_asr, whisper_model_loaded
    if not enable_asr:
        return ""
    
    # Check and wait until model is loaded before running it.
    checks_for_model_loading = 40
    checks_left_for_model_loading = checks_for_model_loading
    sleep_time = 0.2
    while checks_left_for_model_loading>0 and not whisper_model_loaded:
        time.sleep(sleep_time)
        checks_left_for_model_loading -= 1
    assert checks_left_for_model_loading>0, f"Whisper model loading not finished even after {(checks_for_model_loading*sleep_time)} seconds"
    if checks_left_for_model_loading == 0:
        return ""

    new_file_path = process_input_audio(audio_path)
    language = "english"
    if selected_ChatGLM: language = "chinese"
    transcription = decode_audio_file( new_file_path, whisper_model, language=language, mel_filters_dir=asr_assets_path)

    if whisper_model is not None:        
        whisper_model.unload_model()
        del whisper_model
        whisper_model = None
    whisper_model_loaded = False
    return transcription

nvmlInit()
mic_init_handler()
res = mic_recording_done_handler(audio_path)
print(res)