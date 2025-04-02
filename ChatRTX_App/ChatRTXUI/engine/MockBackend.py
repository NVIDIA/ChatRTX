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


import logging.handlers
import time
from configuration import Configuration
import logging
import random
from enum import Enum
from ResponseUtility import getLocalLinksMarkdown, getImagesMarkdown

class Mode(Enum):
    RAG = "RAG"
    AI = "AI"

class Backend:
    def __init__(self, model_setup_dir):
        self._logger = logging.getLogger("[MockBackend]")
        self._logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter( '%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
        self._logger.addHandler(console_handler)
        self.config = Configuration()
        self._logger.info(f"Model init for dir {model_setup_dir}")

    def init_model(self, model_id: str):
        self._rand_handle()
        self._logger.info(f"init_model with model_id={model_id}")

    def ChatRTX(self, chatrtx_mode, data_dir = None):
        self._logger.info(f"ChatRTX called with chatrxt_mode={chatrtx_mode} data_dir={data_dir}")

    def generate_query_engine(self, data_dir):
        self._logger.info(f"generate_query_engine called with {data_dir}")

    def query(self, query):
        self._logger.info(f"query called with query={query}")
        self._rand_handle()
        if self.chatrtx_mode == Mode.AI:
            response = f'AI mode response for query ${query}'
            return response

    def query_stream(self, query):
        self._logger.info(f"query_stream called with query={query}")
        self._rand_handle()
        response = ['Streaming mode ', ' response', ' mocking', ' mode for query ', query]
        for token in response:
            time.sleep(0.2)
            yield str(token)
        yield getLocalLinksMarkdown([
            "C:\\Users\\akushwaha\\workspace\\trt-llm-rag-windows\\ChatRTX\\sample_data\\chinese_dataset\\“传送门：序曲”RTX版和“瑞奇与叮当：时空跳转”的Game-Ready驱动.txt",
            "C:\\Users\\akushwaha\\workspace\\trt-llm-rag-windows\\ChatRTX\\sample_data\\dataset\\alan-wake-2-cyberpunk-2077-phantom-liberty-dlss-3-5.txt",
            "C:\\Users\\akushwaha\\workspace\\trt-llm-rag-windows\\ChatRTX\\sample_data\\dataset\\alan-wake-2-cyberpunk-2077-phantom-liberty-dlss-3-5.txt"

        ])
        yield getImagesMarkdown([
            "C:\\Users\\akushwaha\\workspace\\trt-llm-rag-windows\\ChatRTX\\sample_data\\images_dataset\\Loft1.jpg",
            "C:\\Users\\akushwaha\\workspace\\trt-llm-rag-windows\\ChatRTX\\sample_data\\images_dataset\\Loft1.jpg",
            "C:\\Users\\akushwaha\\workspace\\trt-llm-rag-windows\\ChatRTX\\sample_data\\images_dataset\\Loft1.jpg"
        ]
            
        )

    def set_chatrtx_mode(self, chat_mode: Mode):
        self._logger.info(f"set_chatrtx_model called with chat_mode={chat_mode}")
        if chat_mode == Mode.AI:
            self.chatrtx_mode = Mode.AI
        elif chat_mode == Mode.RAG:
            self.chatrtx_mode = Mode.RAG

        if self.chatrtx_mode == Mode.AI:
            source = "nodataset"
        elif self.chatrtx_mode == Mode.RAG:
            source = "directory"

        success = self._rand_handle()
        if success:
            dataInfo = self.config.get_config('dataset')
            dataInfo['selected'] = source
            self.config.write_default_config('dataset', dataInfo)
        return success

    def set_dataset_path(self, path):
        self._logger.info(f"set_dataset_path called with path={path}")
        success = self._rand_handle()
        if success:
            self.generate_query_engine(path)
            dataInfo = self.config.get_config('dataset')
            dataInfo['selected_path'] = path
            self.config.write_default_config('dataset', dataInfo)
        return success

    def generate_index(self):
        success = self._rand_handle()
        return success

    def download_model(self, model_id):
        self._logger.info(f"download model_id called with model_id={model_id}")
        status = self._rand_handle()
        if status:
            modelInfo = self.config.get_config('models/supported')
            for i in range(len(modelInfo)):
                if modelInfo[i]['id'] == model_id:
                    modelInfo[i]['downloaded'] = True
            self.config.write_default_config('models/supported', modelInfo)
        return status

    def install_model(self, model_id):
        self._logger.info(f"install_model called with model_id={model_id}")
        status = self._rand_handle()
        if status:
            modelInfo = self.config.get_config('models/supported')
            for i in range(len(modelInfo)):
                if modelInfo[i]['id'] == model_id:
                    modelInfo[i]['setup_finished'] = True
            self.config.write_default_config('models/supported', modelInfo)

        self._logger.info(f"Install model status {status}")

        return status

    def delete_model(self, model_id):
        self._logger.info(f"delete_model called with model_id={model_id}")
        success = self._rand_handle()
        if success:
            modelInfo = self.config.get_config('models/supported')
            for i in range(len(modelInfo)):
                if modelInfo[i]['id'] == model_id:
                    modelInfo[i]['setup_finished'] = False
                    modelInfo[i]['downloaded'] = False
            self.config.write_default_config('models/supported', modelInfo)
        return success

    def set_active_model(self, model_id):
        self._logger.info(f"set_active_model called with model_id={model_id}")
        success = self._rand_handle()
        if success:
            self.config.write_default_config('models/selected', model_id)
        return success
    
    def init_asr_model(self):
        self._logger.info(f"init_asr_model_call")
        self._rand_handle()
        # initialize transcription model
        return
    
    def get_text_from_audio(self, audio_path):
        # Return transcribed text via this function
        self._rand_handle()
        return f'Text generated for audio {audio_path}'

    def _rand_handle(self):
        # 1/3rd mocked as success
        time.sleep(5)
        return True if random.randint(1, 12) % 4 != 0 else False