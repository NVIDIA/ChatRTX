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

from ChatRTX.llm_prompt_templates import LLMPromptTemplate
import os, json
from ChatRTX.logger import ChatRTXLogger
import logging

class ChatRTX:
    """
    Manages operations on language models including initialization, and response generation.
    """
    # Class-level constants for directory and file names
    ENGINE_DIR = "engine"
    ENGINE_NAME = "engine"
    MODEL_DIR = "model"
    TOKENIZER_DIR = "tokenizer_local_dir"
    VOCAB_DIR = "vocab_local_dir"
    VOCAB_FILE_KEY = "vocab_file"

    def __init__(self, models_info_map, model_download_dir, use_nims = True):
        """
        Initialize the ChatRTX object.

        :param models_info_map: A list of dictionaries containing model information.
        """
        self.use_nims = use_nims
        self._models_info_map = models_info_map
        self._model_directory = os.path.join(model_download_dir, "models")
        self._llm = None

        # Initialize the logger
        ChatRTXLogger(log_level=logging.INFO, log_file='ChatRTX.log')
        self._logger = ChatRTXLogger.get_logger()
        app_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config/app_config.json")
        self._app_config_info = self._load_config(app_config)

    def init_llm_model(self, model_id, **kwqags):
        """
        Initialize the language model based on the provided model ID and backend.

        :param model_id: The ID of the model to initialize.
        :param backend: The backend to use for the model. Default is "TRTLLM".
        :return: True if initialization is successful, False otherwise.
        """
        model_info_list = self._models_info_map
        model_info = next((m for m in model_info_list if m["id"] == model_id), None)
        if model_info is None:
            self._logger.error(f"Model {model_id} not found.")
            return False
        backend = model_info["backend"]
        try:

            if backend == "nims":
                self.use_nims = True
                force_nim_start = kwqags['force_nim_start']
                # start the Load the NIM
                from ChatRTX.model_manager.nim_manager import NIMManager
                self.nim_manager = NIMManager.get_instance()
                ports = []
                grpc_ports = []
                print(model_info)
                nim_profile = model_info["nims_id"]
                self.nim_id = model_info["id"]
                self._active_nim = nim_profile
                if self.nim_manager.start_nim(nim_profile, http_ports=ports, grpc_ports=grpc_ports, force_nim_start=force_nim_start) == False:
                    self._logger.error(f"Start NIM fail for NIm if {nim_profile}")
                    return False
                from openai import OpenAI
                self._llm = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used") # TODO, check the port number
                return True
            else:
                # Find the model information in the internal map using the provided model_id
                from ChatRTX.inference.trtllm.trtllm import TrtLlm
                if backend != "TRTLLM":
                    self._logger.error(f"Unsupported backend '%s'. Currently, only 'TRTLLM' is supported.", backend)
                    raise ValueError(f"Unsupported backend '{backend}'. Currently, only 'TRTLLM' is supported.")
                self.use_nims = False
                # Construct paths for model components
                model_path = os.path.join(self._model_directory, model_info["id"], ChatRTX.ENGINE_DIR)
                engine_file_path = os.path.join(model_path, model_info["metadata"].get(ChatRTX.ENGINE_NAME, ""))
                tokenizer_dir = os.path.join(self._model_directory, model_info["id"],
                                             model_info.get("prerequisite", {}).get(ChatRTX.TOKENIZER_DIR, "")) \
                    if ("tokenizer_local_dir" in model_info["prerequisite"]) else None

                vocab_file = os.path.join(self._model_directory, model_info["id"],
                                          model_info["prerequisite"][ChatRTX.VOCAB_DIR],
                                          model_info["prerequisite"]["tokenizer_files"]["vocab_file"]) \
                    if (ChatRTX.VOCAB_DIR in model_info["prerequisite"]) else None
                # Log the constructed paths
                self._logger.debug("Model path: %s", model_path)
                self._logger.debug("Engine file path: %s", engine_file_path)
                self._logger.debug("Tokenizer directory: %s", tokenizer_dir)
                self._logger.debug("Vocab file: %s", vocab_file)

                # read the app config file to init the default values
                use_py_session = kwqags['use_py_session'] if 'use_py_session' in kwqags else self._app_config_info[
                    'use_py_session']
                add_special_tokens = kwqags['add_special_tokens'] if 'add_special_tokens' in kwqags else \
                self._app_config_info['add_special_tokens']
                trtLlm_debug_mode = kwqags['trtLlm_debug_mode'] if 'trtLlm_debug_mode' in kwqags else \
                self._app_config_info['trtLlm_debug_mode']
                # Initialize the TrtLlm object
                self._llm = TrtLlm(
                    model_path=model_path,
                    tokenizer_dir=tokenizer_dir,
                    temperature=model_info["metadata"].get("temperature", 0.1),
                    max_new_tokens=model_info["metadata"].get("max_new_tokens", None),
                    context_window=model_info["metadata"].get("max_input_token", None),
                    vocab_file=vocab_file,
                    use_py_session=use_py_session,
                    add_special_tokens=add_special_tokens,
                    trtLlm_debug_mode=trtLlm_debug_mode
                )
                return True
        except Exception as e:
            self._logger.error(f"Failed to init TRTLLM model object: Error {str(e)}")
            return False

    def init_clip_model(self, model_id):
        """
        Initialize the language model based on the provided model ID and backend.

        :param model_id: The ID of the model to initialize.
        :param backend: The backend to use for the model. Default is "TRTLLM".
        :return: True if initialization is successful, False otherwise.
        """
        try:
            # Find the model information in the internal map using the provided model_id
            from ChatRTX.inference.pytorch.CLIP import ClipInference
            model_info = next((info for info in self._models_info_map if info["id"] == model_id), None)
            if model_info is None:
                self._logger.error("Model ID '%s' not found in the configuration.", model_id)
                raise ValueError(f"Model ID '{model_id}' not found in the configuration.")

            # Construct paths for model components
            model_path = os.path.join(self._model_directory, model_info["id"])

            self.clip_inference = ClipInference()
            status = self.clip_inference.load_model(model_path)
            return status
        except Exception as e:
            self._logger.error(f"Failed to init clip model object: Error {str(e)}")
            return False

    def generate_clip_engine(self, image_data_dir, force_rewrite=True):
        if self.clip_inference is not None:
            return self.clip_inference.generate_clip_engine(image_data_dir, force_rewrite=True)
        else:
            self._logger.error("No clip inferance object")
            return False

    def generate_clip_response(self, input_text, top_matches_path, min_clip_score):
        if self.clip_inference is not None:
            return self.clip_inference.generate_clip_response(input_text, top_matches_path, min_clip_score)
        else:
            self._logger.error("No clip inferance object")
            return False

    def generate_response(self, query):
        """
        Generate a response for a given query using the loaded language model.

        :param query: The query string for which to generate a response.
        :return: The generated response.
        :raises Exception: If no model is loaded or if response generation fails.
        """
        if self._llm is None:
            self._logger.error("No model is loaded. Please load a model before generating responses")
            raise Exception("No model is loaded. Please load a model before generating responses.")

        try:
            if self.use_nims == True:

                messages = [
                    {"role": "user", "content": "Hello! How are you?"},
                    {"role": "assistant", "content": "Hi! I am quite well, how can I help you today?"},
                ]
                messages.append({"role": "user", "content": query})

                chat_response = self._llm.chat.completions.create(
                    model=self.nim_id,
                    messages=messages,
                    max_tokens=2000,
                    stream=False
                )

                # Check if choices are available
                if not chat_response.choices:
                    self._logger.error("No choices returned in the chat response.")
                    raise Exception("No response generated by the model.")

                # Access the content attribute correctly
                assistant_message = chat_response.choices[0].message.content
                return assistant_message
            else:
                # Create a prompt template and generate the prompt
                prompt_template = LLMPromptTemplate()
                prompt = prompt_template.model_default_template(model=self._llm.get_model_name(), query=query)
                # Generate and return the response using the language model
                return self._llm.complete(prompt)
        except Exception as e:
            self._logger.error(f"Failed to generate the response: Error: {str(e)}")
            raise Exception(f"Failed to generate the response: {str(e)}")

    def generate_stream_response(self, query):
        """
        Generate a streaming response for a given query using the loaded language model.

        :param query: The query string for which to generate a streaming response.
        :raises Exception: If no model is loaded or if streaming response generation fails.
        """
        if self._llm is None:
            self._logger.error("No model is loaded. Please load a model before generating responses")
            raise Exception("No model is loaded. Please load a model before generating responses.")

        try:
            if self.use_nims == True:
                messages = [
                    {"role": "user", "content": "Hello! How are you?"},
                    {"role": "assistant", "content": "Hi! I am quite well, how can I help you today?"},
                ]
                messages.append({"role": "user", "content": query})
                chat_response = self._llm.chat.completions.create(
                    model=self.nim_id,
                    messages=messages,
                    max_tokens=2000,
                    stream=True  # Enable streaming
                )

                for chunk in chat_response:
                    content = chunk.choices[0].delta.content or ""
                    yield content
            else:
                # Create a prompt template and generate the prompt
                prompt_template = LLMPromptTemplate()
                prompt = prompt_template.model_default_template(model=self._llm.get_model_name(), query=query)

                # Generate and print the streaming response using the language model
                response_tokens = self._llm.stream_complete(prompt)
                total = ""
                for response in response_tokens:
                    yield response

        except Exception as e:
            self._logger.error(f"Failed to generate the stream response: Error {str(e)}")
            raise Exception(f"Failed to generate the stream response: {str(e)}")

    def unload_llm(self):
        """
        Unload the currently loaded language model, if any.

        :raises Exception: If unloading the model fails.
        """
        if self._llm is not None:
            try:
                # Unload the language model
                if not self.use_nims:
                    self._llm.unload_llm()
                else:
                    self.nim_manager.stop_nim(self._active_nim)
                    self._active_nim = None
                self._llm = None
                self._logger.info("Language model unloaded successfully.")
            except Exception as e:
                self._logger.error(f"Failed to unload the language model: Error {str(e)}")
                raise Exception(f"Failed to unload the language model: {str(e)}")

    def unload_trt_llm(self):
        if self._llm is not None:
            try:
                # Unload the language model
                if not self.use_nims:
                    self._llm.unload_llm()
                    self._llm = None
                    return True
            except Exception as e:
                self._logger.error(f"Failed to unload the language model: Error {str(e)}")
                raise Exception(f"Failed to unload the language model: {str(e)}")


    def _load_config(self, file_name):
        """
        Loads the configuration from the specified file.

        Args:
            file_name (str): The name of the configuration file.

        Returns:
            dict: A dictionary containing the supported models information.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If there is an error decoding the JSON.
            Exception: If an unexpected error occurs.
        """
        try:
            with open(file_name, 'r', encoding='utf8') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The configuration file {file_name} was not found.")
        except json.JSONDecodeError:
            raise ValueError(f"There was an error decoding the JSON from the file {file_name}.")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
