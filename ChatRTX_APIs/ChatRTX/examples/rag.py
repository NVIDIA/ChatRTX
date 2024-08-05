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

from ChatRTX.chatrtx_rag import ChatRTXRag
from ChatRTX.model_manager.model_manager import ModelManager
import gc
import torch
import logging
import sys
from ChatRTX.logger import ChatRTXLogger

# Initialize logger
ChatRTXLogger(log_level=logging.INFO)

# Define the directory where models will be downloaded
model_download_dir = "C:\\ProgramData\\NVIDIA Corporation\\ChatRTX"

# Initialize the model manager with the specified download directory
model_manager = ModelManager(model_download_dir)

# Get the list of available models and print it
model_list = model_manager.get_model_list()
print(f"Model list: {model_list}")

def handle_model_setup(model_id):
    # Check if the specified model is downloaded, if not, download it
    if not model_manager.is_model_downloaded(model_id):
        status = model_manager.download_model(model_id)
        if not status:
            logging.error(f"Model download failed for the model: {model_id}")
            sys.exit(1)

    # Check if the specified model is installed, if not, install it
    if not model_manager.is_model_installed(model_id):
        print("Building TRT-LLM engine....")
        status = model_manager.install_model(model_id)
        if not status:
            logging.error(f"Model installation failed for the model: {model_id}")
            sys.exit(1)

# Define the model ID and handle setup
model_id = "llama2_13b_AWQ_INT4_chat"
handle_model_setup(model_id)

# Get the information about the models
model_info = model_manager.get_model_info()

# Initialize the ChatRTXRag object with the model information and download directory
chat_rtx_rag = ChatRTXRag(model_info, model_download_dir)

try:
    # Initialize the LlamaIndex LLM model with the specified model ID
    status = chat_rtx_rag.init_llamaIndex_llm(model_id)
    if not status:
        logger.error(f"Failed to load the model: {model_id}")
        sys.exit(1)

    # Set the embedding model
    chat_rtx_rag.set_embedding_model("BAAI/bge-small-en-v1.5", 384)

    # Set the RAG settings
    chat_rtx_rag.set_rag_setting(chunk_size=512, chunk_overlap=200)

    # Generate a query engine for the specified data directory
    engine = chat_rtx_rag.generate_query_engine("../data")

    while True:
        # Take user query
        query = input("Enter your query (type 'exit' to quit): ")

        # Check if the user wants to exit
        if query.lower() == 'exit':
            print("Exiting the application.")
            break

        # Generate a response to the query using the query engine
        try:
            answer = chat_rtx_rag.generate_response(query, engine)
            print(f"Answer: {answer}")
        except Exception as e:
            logging.error(f"Unable to generate a response: {str(e)}")

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")

finally:
    # Clear the CUDA cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()

    # Unload the current LLM model
    chat_rtx_rag.unload_llm()
