# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time

import gradio as gr
import argparse
from trt_llama_api import TrtLlmAPI #llama_index does not currently support TRT-LLM. The trt_llama_api.py file defines a llama_index compatible interface for TRT-LLM.
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import set_global_service_context
from faiss_vector_storage import FaissEmbeddingStorage

# Create an argument parser
parser = argparse.ArgumentParser(description='NVIDIA Chatbot Parameters')

# Add arguments
parser.add_argument('--trt_engine_path', type=str, required=True,
                    help="Path to the TensorRT engine.", default="")
parser.add_argument('--trt_engine_name', type=str, required=True,
                    help="Name of the TensorRT engine.", default="")
parser.add_argument('--tokenizer_dir_path', type=str, required=True,
                    help="Directory path for the tokenizer.", default="")
parser.add_argument('--embedded_model', type=str,
                    help="Name or path of the embedded model. Defaults to 'sentence-transformers/all-MiniLM-L6-v2' if "
                         "not provided.",
                    default='sentence-transformers/all-MiniLM-L6-v2')
parser.add_argument('--data_dir', type=str, required=False,
                    help="Directory path for data.", default="./dataset")
parser.add_argument('--verbose', type=bool, required=False,
                    help="Enable verbose logging.", default=False)
# Parse the arguments
args = parser.parse_args()

# Use the provided arguments
trt_engine_path = args.trt_engine_path
trt_engine_name = args.trt_engine_name
tokenizer_dir_path = args.tokenizer_dir_path
embedded_model = args.embedded_model
data_dir = args.data_dir
verbose = args.verbose

# create trt_llm engine object
llm = TrtLlmAPI(
    model_path=trt_engine_path,
    engine_name=trt_engine_name,
    tokenizer_dir=tokenizer_dir_path,
    temperature=0.1,
    max_new_tokens=1024,
    context_window=3900,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False
)

# create embeddings model object
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embedded_model))
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

# load the vectorstore index
faiss_storage = FaissEmbeddingStorage(data_dir=data_dir)
query_engine = faiss_storage.get_query_engine()

# chat function to trigger inference
def chatbot(query, history):
    if verbose:
        start_time = time.time()
        response = query_engine.query(query)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Inference e2e time    : {elapsed_time:.2f} seconds \n")

    else:
        response = query_engine.query(query)
    return str(response)

# Gradio UI inference function
interface = gr.ChatInterface(
    fn=chatbot,                        # Function to call on user input
    title="Chat with GeForce News",    # Title of the web page
    description="Ask me anything!",    # Description
)
interface.launch(server_name="localhost")
