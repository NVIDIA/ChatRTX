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

from enum import Enum
import threading
import json
from configuration import Configuration
import os
import time
import logging

RUN_MOCK = os.environ.get('MOCK')

if not RUN_MOCK:
    from backend import Backend, Mode
else:
    from MockBackend import Backend, Mode

class Model(Enum):
    MISTRAL = 'Mistral 7B int4'

class Events(Enum):
    NONE = 'NONE'
    ON_PYTHON_ENGINE_INIT = 'ON_PYTHON_ENGINE_INIT'
    ON_PYTHON_ENGINE_INIT_ERROR = 'ON_PYTHON_ENGINE_INIT_ERROR'
    ON_INIT_APP = 'ON_INIT_APP'
    ACTIVE_MODEL_UPDATE = 'ACTIVE_MODEL_UPDATE'
    ACTIVE_MODEL_UPDATE_ERROR = 'ACTIVE_MODEL_UPDATE_ERROR'
    MODEL_DOWNLOAD_ERROR = 'MODEL_DOWNLOAD_ERROR'
    MODEL_INSTALL_ERROR = 'MODEL_INSTALL_ERROR'
    MODEL_DELETE_ERROR = 'MODEL_DELETE_ERROR'
    MODEL_DOWNLOADED = 'MODEL_DOWNLOADED'
    MODEL_INSTALLED = 'MODEL_INSTALLED'
    MODEL_DELETED = 'MODEL_DELETED'
    ON_DATASET_UPDATE = 'ON_DATASET_UPDATE'
    ON_DATASET_UPDATE_ERROR = 'ON_DATASET_UPDATE_ERROR'
    ON_INDEX_REGENERATED = 'ON_DATA_REGENERATED'
    ON_INDEX_REGENERATE_ERROR = 'ON_DATA_REGENERATE_ERROR'
    ON_BASE_MODEL_DOWNLOADED = 'ON_BASE_MODEL_DOWNLOADED'
    ON_BASE_MODEL_DOWNLOAD_ERROR = 'ON_BASE_MODEL_DOWNLOAD_ERROR'
    ON_PROFILE_CREATED = 'ON_PROFILE_CREATED'
    ON_PROFILE_CREATE_ERROR = 'ON_PROFILE_CREATE_ERROR'
    ON_PROFILE_SELECTED = 'ON_PROFILE_SELECTED'
    ON_PROFILE_SELECT_ERROR = 'ON_PROFILE_SELECT_ERROR'
    ON_PROFILE_DELETED = 'ON_PROFILE_DELETED'
    ON_PROFILE_DELETED_ERROR = 'ON_PROFILE_DELETED_ERROR'
    ON_FINE_TUNING_ENABLED = 'ON_FINE_TUNING_ENABLED'
    ON_FINE_TUNING_ENABLE_ERROR = 'ON_FINE_TUNING_ENABLE_ERROR'
    ON_FINE_TUNING_DISABLED = 'ON_FINE_TUNING_DISABLED'
    ON_FINE_TUNING_DISABLE_ERROR = 'ON_FINE_TUNING_DISABLE_ERROR'


class ChatBot:
    _eventEmitter = None
    root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
    data_path = "%programdata%\\NVIDIA Corporation\\chatrtx"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.data_path = self._expand_programdata_path(self.data_path)
        self.model = Model.MISTRAL
        self.history = None
        self.ready = True
        self.model_info = None
        self._logger = logging.getLogger("[ChatRTXUIEngine]")
        self._logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter( '%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
        self._logger.addHandler(console_handler)
        self.ft_model = None


    def _determine_mode(self) -> Mode:
        source = self.config.get_config('dataset/selected')
        if source == "directory":
            return Mode.RAG
        elif source == "nodataset":
            return Mode.AI
        else:
            raise ValueError(f"Invalid mode: {source}. Mode must be either 'directory' or 'nodataset'.")

    def _setup_dataset_dir(self) -> str:
        dataset_dir = self.config.get_config('dataset/selected_path')
        if not os.path.isabs(dataset_dir):
            dataset_dir = os.path.abspath(dataset_dir)
        else:
            dataset_dir = os.path.normpath(dataset_dir)
        return dataset_dir
        
    def _expand_programdata_path(self, path):
        expanded_path = os.path.expandvars(path)
        return expanded_path

    def isReady(self):
        return self.ready

    def _handle_init_chatbot_engine(self):
        
        try:
            self.backend = Backend(model_setup_dir=self.data_path)
            self.config = Configuration()
            self.chatrtx_mode = self._determine_mode()
            self.dataset_dir = self._setup_dataset_dir()
            self._logger.info(
                f"Application started with \n Dataset path : {self.dataset_dir} \n App mode {self.chatrtx_mode}")
        except Exception as e:
            self._logger.error(f"Error in setting up the data folder path: {e}")
            return False       
        
        
        try:
            # Retrieve the selected model ID from the configuration
            model_id = self.config.get_config('models/selected')

            # Initialize the backend model with the specified model ID
            self.backend.init_model(model_id=model_id)

            # Initialize ChatRTX based on the selected mode (AI or RAG)
            if self.chatrtx_mode == Mode.AI:
                status = self.backend.ChatRTX(chatrtx_mode=self.chatrtx_mode)
            elif self.chatrtx_mode == Mode.RAG:
                status = self.backend.ChatRTX(chatrtx_mode=self.chatrtx_mode, data_dir=self.dataset_dir)
            else:
                raise ValueError(f"Invalid mode: {self.chatrtx_mode}. Mode must be either 'AI' or 'RAG'.")

            return status
        except Exception as e:
            self._logger.error(f"Error while loading model: {e}")
            return False

    def init_chatbot_engine(self, session_id):
        assert self.session_id == session_id
        return self._handle_with_condition(lambda: self._handle_init_chatbot_engine(), Events.ON_PYTHON_ENGINE_INIT,
                                           Events.ON_PYTHON_ENGINE_INIT_ERROR)

    def _handle(self, handler, eventName: Events, data = None):
        def task():
            handler()
            self.send_event(eventName, json.dumps(data))

        t1 = threading.Thread(target=task)
        t1.start()

    def _handle_with_condition(self, handler, successEventName: Events, failureEventName: Events, eventData=None):
        def task():
            data = handler()
            try:
                if data:
                    self.send_event(successEventName, json.dumps(eventData))
                else:
                    self.send_event(failureEventName, json.dumps(eventData))
            except:
                self.send_event(failureEventName, json.dumps(eventData))

        t1 = threading.Thread(target=task)
        t1.start()
        return t1

    def query(self, query: str, is_streaming: bool, session_id: str):
        assert self.session_id == session_id
        try:
            if self.backend is not None:
                answer = self.backend.query_stream(query=query)
                for token in answer:
                    yield token
            else:
                self._logger.warning("Backend is not initialized.")
        except Exception as e:
            self._logger.error(f"Error during query execution: {e}")
            yield "Problem generating response: Data source may be empty or unsupported – Ensure dataset compatibility with the AI model"


    def generate_index(self, session_id: str):
        assert self.session_id == session_id
        return self._handle_with_condition(lambda: self.backend.generate_index(), Events.ON_INDEX_REGENERATED,
                                    Events.ON_INDEX_REGENERATE_ERROR)

    def set_dataset_source(self, source: Mode, session_id: str):
        assert self.session_id == session_id
        if source == "nodataset":
            self.chatrtx_mode = Mode.AI
        else:
            self.chatrtx_mode = Mode.RAG
        return self._handle_with_condition(lambda: self.backend.set_chatrtx_mode(self.chatrtx_mode), Events.ON_DATASET_UPDATE,
                                    Events.ON_DATASET_UPDATE_ERROR)

    def set_dataset_path(self, path, session_id):
        assert self.session_id == session_id
        self.dataset_dir = path
        return self._handle_with_condition(lambda: self.backend.set_dataset_path(path), Events.ON_DATASET_UPDATE,
                                    Events.ON_DATASET_UPDATE_ERROR)


    def set_active_model(self, model_id, session_id: str):
        self._logger.info(f"Set active modelid {model_id}")
        assert self.session_id == session_id
        return self._handle_with_condition(lambda: self.backend.set_active_model(model_id), Events.ACTIVE_MODEL_UPDATE,
                                    Events.ACTIVE_MODEL_UPDATE_ERROR, model_id)

    def shutdown(self, session_id: str):
        assert self.session_id == session_id
        self.app.on_shutdown_handler(session_id)
        return True

    def set_emitter(self, eventEmitter):
        self._eventEmitter = eventEmitter

    def send_event(self, eventName: Events, data):
        try:
            self._eventEmitter(eventName.value, data)
        except:
            print("An exception occurred while sending event")

    def init_asr_model(self, session_id):
        assert self.session_id == session_id
        self.backend.init_asr_model()
        return True

    def get_text_from_audio(self, audio_path, session_id):
        assert self.session_id == session_id
        return self.backend.get_text_from_audio(audio_path)

    def download_model(self, model_id, session_id):
        assert self.session_id == session_id
        self._logger.info(f"Print the model id {model_id}")
        return self._handle_with_condition(lambda: self.backend.download_model(model_id), Events.MODEL_DOWNLOADED,
                                    Events.MODEL_DOWNLOAD_ERROR, model_id)

    def install_model(self, model_id, session_id):
        assert self.session_id == session_id
        self._logger.info(f"BUild engine for model: {model_id}")
        return self._handle_with_condition(lambda: self.backend.install_model(model_id), Events.MODEL_INSTALLED,
                                    Events.MODEL_INSTALL_ERROR, model_id)


    def delete_model(self, model_id, session_id):
        assert self.session_id == session_id
        self._handle_with_condition(lambda: self.backend.delete_model(model_id), Events.MODEL_DELETED,
                                    Events.MODEL_DELETE_ERROR, model_id)
        return True

    def get_text_from_audio(self, audio_path, session_id):
        assert self.session_id == session_id
        return self.backend.get_text_from_audio(audio_path)

    def get_dataset_info(self, session_id):
        assert self.session_id == session_id
        val = self.config.get_config('dataset')
        retVal = {
            'sources': val['sources'],
            'selected': val['selected'],
            'path': val['path'],
            'path_chinese': val['path_chinese'],
            'path_clip': val['path_clip'],
            'selected_path': val['selected_path']
        }
        return json.dumps(retVal)

    def get_model_info(self, session_id):
        assert self.session_id == session_id
        self.model_info = self.config.get_config('models')
        return json.dumps(self.model_info)
    
    def get_sample_question_info(self, session_id):
        assert self.session_id == session_id
        sample_question_info = self.config.get_config('sample_questions')
        sample_question_info['default']['dataset_path'] = sample_question_info['default']['dataset_path']
        sample_question_info['chinese']['dataset_path'] = sample_question_info['chinese']['dataset_path']
        sample_question_info['images']['dataset_path'] = sample_question_info['images']['dataset_path']
        return json.dumps(sample_question_info)

    def fake_handle(self):
        time.sleep(5)
        return True

    def get_fine_tunning_info(self, session_id):
        assert self.session_id == session_id
        val = self.config.get_config('models/fine_tune_data')
        return json.dumps(val)
