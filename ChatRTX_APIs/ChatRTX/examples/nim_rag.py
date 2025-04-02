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

import os
import faiss
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from ChatRTX.model_manager.nim_manager import NIMManager

def initialize_nim_manager(nim_profile):
    nim_manager = NIMManager()
    if not nim_manager.download_nim(nim_profile):
        print(f"Could not download NIM {nim_profile}, exiting")
        exit()
    
    http_ports = []
    grpc_ports = []
    if not nim_manager.start_nim(nim_profile, http_ports=http_ports, grpc_ports=grpc_ports, force_nim_start=True):
        print(f"Start NIM failed for {nim_profile}")
        exit()
    assert len(http_ports) == 1 and http_ports[0] == '8000' and len(grpc_ports)==0, "Example only provided for HTTP port 8000."
    return nim_manager

def setup_index(data_path, model_name):
    Settings.llm = NVIDIA(base_url="http://localhost:8000/v1", model=model_name)
    embedding_model_name = "intfloat/multilingual-e5-base"
    embedding_model_dimension = 768
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=200)

    documents = SimpleDirectoryReader(data_path).load_data()

    faiss_index = faiss.IndexFlatL2(embedding_model_dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index.as_query_engine(similarity_top_k=4, streaming=True)

def main():

    # Update to path of folder having RAG dataset (text, pdf, doc/docx)
    data_path = "D:\\trt-llm-rag-windows\\ChatRTX_APIs\\ChatRTX\\sample_data\\dataset"

    print("Welcome to RAG using ChatRTX APIs example!")
    print("Type your query and press Enter.")
    print("Type 'exit' or 'quit' to terminate the program.\n")
    
    nim_profile = "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.0-RTX"
    model_name = "meta/llama-3.1-8b-instruct"
    
    try:
        nim_manager = initialize_nim_manager(nim_profile)
        query_engine = setup_index(data_path, model_name)
        
        while True:
            user_query = input("You: ").strip()
            if user_query.lower() in {'exit', 'quit'}:
                print("Exiting the program. Goodbye!")
                break
            
            if not user_query:
                print("Please enter a valid query.")
                continue
            
            response = query_engine.query(user_query)
            for text in response.response_gen:
                print(text, end='', flush=True)
                
    except Exception as e:
        print(f"An error occurred: {e}\n")
    finally:
        if 'nim_manager' in locals():
            nim_manager.stop_nim(nim_profile)

if __name__ == "__main__":
    main()

