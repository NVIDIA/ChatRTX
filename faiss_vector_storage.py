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

import faiss, os
from llama_index.vector_stores import FaissVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index import StorageContext, load_index_from_storage
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.storage.index_store.simple_index_store import SimpleIndexStore


class FaissEmbeddingStorage:

    def __init__(self, data_dir, dimension=384):
        """        Initialize the object with the given data directory and dimension.

        Args:
            data_dir (str): The directory path where the data is located.
            dimension (int?): The dimension size. Defaults to 384.
        """

        self.d = dimension
        self.data_dir = data_dir
        self.index = self.initialize_index()

    def initialize_index(self):
        """        Initialize the index for vector storage.

        If the "storage-default" directory exists and is not empty, the function loads the index from the persisted storage
        and returns it. If the directory does not exist or is empty, the function generates new values, creates a new index,
        and persists it in the "storage-default" directory.

        Returns:
            VectorStoreIndex: The initialized or loaded vector store index.
        """

        if os.path.exists("storage-default") and os.listdir("storage-default"):
            print("Using the persisted value")
            vector_store = FaissVectorStore.from_persist_dir("storage-default")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir="storage-default"
            )
            index = load_index_from_storage(storage_context=storage_context)
            return index
        else:
            print("generating new values")
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            faiss_index = faiss.IndexFlatL2(self.d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            index.storage_context.persist(persist_dir = "storage-default")
            return index

    def get_query_engine(self):
        """        Returns a query engine for the index.

        This method returns a query engine for the index, allowing for efficient querying of the data stored in the index.

        Returns:
            QueryEngine: A query engine for the index.
        """

        return self.index.as_query_engine()
