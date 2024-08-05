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

from ChatRTX.chatrtx import ChatRTX
from ChatRTX.model_manager.model_manager import ModelManager
import logging
import sys
from ChatRTX.logger import ChatRTXLogger

# Initialize logger
ChatRTXLogger(log_level=logging.INFO)
logger = ChatRTXLogger.get_logger()

# Define the directory where models will be downloaded
model_download_dir = "C:\\ProgramData\\NVIDIA Corporation\\ChatRTX"

# Initialize the model manager with the specified download directory
model_manager = ModelManager(model_download_dir)

# Get the list of available models and print it
model_list = model_manager.get_model_list()
print(f"Model list: {model_list}")

# Define the model ID, get it from the model_list
model_id = "mistral_7b_AWQ_int4_chat"

# Check if the specified model is already downloaded
if not model_manager.is_model_downloaded(model_id):
    # Download the model if it is not already downloaded
    status = model_manager.download_model(model_id)
    if not status:
        logger.error(f"Model download failed for the model: {model_id}")
        sys.exit(1)

# Check if the specified model is installed
if not model_manager.is_model_installed(model_id):
    # Install the model if it is not already installed
    print("Building TRT-LLM engine....")
    status = model_manager.install_model(model_id)
    if not status:
        logger.error(f"Model installation failed for the model: {model_id}")
        sys.exit(1)

# Get the information about the model
model_info = model_manager.get_model_info()

# Initialize the ChatRTX object with the model information and download directory
chat_rtx = ChatRTX(model_info, model_download_dir)

# Initialize the LLM model with the specified model ID
status = chat_rtx.init_llm_model(model_id, add_special_tokens=True, use_py_session=True)
if not status:
    logger.error(f"Failed to load the model: {model_id}")
    sys.exit(1)

try:
    while True:
        # Take user query
        query = input("Enter your query (type 'exit' to quit): ")

        # Check if the user wants to exit
        if query.lower() == 'exit':
            print("Exiting the application.")
            break

        # Generate a response to the query
        try:
            answer = chat_rtx.generate_stream_response(query)
            total = ""
            for token in answer:
                total = total + token
                print(f"Answer: {total}")
        except Exception as e:
            logger.error(f"Unable to generate a response: {str(e)}")

finally:
    # Unload the LLM model
    chat_rtx.unload_llm()
