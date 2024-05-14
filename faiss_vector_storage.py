# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import faiss
import os
import shutil
import gc
import torch
from llama_index.vector_stores import FaissVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index import StorageContext, load_index_from_storage

class FaissEmbeddingStorage:
    def __init__(self, data_dir, dimension):
        self.d = dimension
        self.data_dir = data_dir
        self.engine = None
        self.persist_dir = f"{self.data_dir}_vector_embedding"

    def initialize_index(self, force_rewrite=False):
        # Check if the persist directory exists and delete it if force_rewrite is true
        if force_rewrite and os.path.exists(self.persist_dir):
            print("Deleting existing directory for a fresh start.")
            self.delete_persist_dir()

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print("Using the persisted value form " + self.persist_dir)
            vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=self.persist_dir
            )
            self.index = load_index_from_storage(storage_context=storage_context)
        else:
            print("Generating new values")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                if os.path.exists(self.data_dir) and os.listdir(self.data_dir):
                    file_metadata = lambda x: {"filename": x}
                    documents = SimpleDirectoryReader(self.data_dir, file_metadata=file_metadata,
                                                    recursive=True, required_exts= [".pdf", ".doc", ".docx", ".txt", ".xml"]).load_data()
                else:
                    print("No files found in the directory. Initializing an empty index.")
                    documents = []
            except Exception as e:
                 documents = []
                 print(f"Error occurred while initializing index: {str(e)}")
            faiss_index = faiss.IndexFlatL2(self.d)
            #faiss_index = faiss.IndexFlatIP(self.d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context ,show_progress = True)
            index.storage_context.persist(persist_dir=self.persist_dir)
            self.index = index
            torch.cuda.empty_cache()
            gc.collect()

    def delete_persist_dir(self):
        if os.path.exists(self.persist_dir) and os.path.isdir(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
            except Exception as e:
                print(f"Error occurred while deleting directory: {str(e)}")

    def get_engine(self,is_chat_engine ,streaming , similarity_top_k):

        if is_chat_engine == True:
            self.engine = self.index.as_chat_engine(
                chat_mode="condense_question",
                streaming=streaming,
                similarity_top_k = similarity_top_k
            )
        else:
            query_engine = self.index.as_query_engine(
                streaming=streaming,
                similarity_top_k = similarity_top_k,
            )
            self.engine = query_engine
        return self.engine

    def reset_engine(self, engine):
        engine.reset()
