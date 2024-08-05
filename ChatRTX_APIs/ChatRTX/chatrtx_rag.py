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

from ChatRTX.rags.llama_index.trtllm_api import TrtLlmAPI
from ChatRTX.inference.trtllm.utils import (read_model_name)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from ChatRTX.llm_prompt_templates import LLMPromptTemplate
import faiss
import os, json
import gc, torch
from ChatRTX.logger import ChatRTXLogger
import shutil
import logging
class ChatRTXRag:
    """
    Manages operations on language models including initialization, and response generation.
    """
    ENGINE_DIR = "engine"
    ENGINE_NAME = "engine"
    MODEL_DIR = "model"
    TOKENIZER_DIR = "tokenizer_local_dir"
    VOCAB_DIR = "vocab_local_dir"
    VOCAB_FILE_KEY = "vocab_file"

    def __init__(self, models_info_map, model_download_dir):
        """
        Initialize the ChatRTXRag object.

        :param models_info_map: A list of dictionaries containing model information.
        """
        self._models_info_map = models_info_map
        self._model_directory = os.path.join(model_download_dir, "models")
        self._llm = None
        self._embedding_model = None
        self._embedding_dim = None
        ChatRTXLogger(log_level=logging.INFO, log_file='chatRTX.log')
        self._logger = ChatRTXLogger.get_logger()
        self._logger.info("ChatRTX RAG mode initialized with model directory: %s", self._model_directory)
        app_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config/app_config.json")
        self._app_config_info = self._load_config(app_config)

    def init_llamaIndex_llm(self, model_id, backend="TRTLLM", **kwargs):
        """
        Initialize the LlamaIndex language model based on the provided model ID and backend.

        :param model_id: The ID of the model to initialize.
        :param backend: The backend to use for the model. Default is "TRTLLM".
        :return: True if initialization is successful, False otherwise.
        :raises ValueError: If the model ID is not found or the backend is unsupported.
        """

        try:
            # Find the model information in the internal map using the provided model_id
            model_info = next((info for info in self._models_info_map if info["id"] == model_id), None)
            if model_info is None:
                raise ValueError(f"Model ID '{model_id}' not found in the configuration.")

            if backend != "TRTLLM":
                raise ValueError(f"Unsupported backend '{backend}'. Currently, only 'TRTLLM' is supported.")

            model_path = os.path.join(self._model_directory, model_info["id"], ChatRTXRag.ENGINE_DIR)
            engine_file_path = os.path.join(model_path, model_info["metadata"].get(ChatRTXRag.ENGINE_NAME, ""))
            tokenizer_dir = os.path.join(self._model_directory, model_info["id"],
                                         model_info.get("prerequisite", {}).get(ChatRTXRag.TOKENIZER_DIR, "")) if ("tokenizer_local_dir"
                                                                                                                    in model_info["prerequisite"]) else None

            vocab_file = os.path.join(self._model_directory, model_info["id"], model_info["prerequisite"][ChatRTXRag.VOCAB_DIR],
                                      model_info["prerequisite"]["tokenizer_files"]["vocab_file"] ) if (ChatRTXRag.VOCAB_DIR
                                                                                                           in model_info["prerequisite"]) else None

            self._logger.debug("Model path: %s", model_path)
            self._logger.debug("Engine file path: %s", engine_file_path)
            self._logger.debug("Tokenizer directory: %s", tokenizer_dir)
            self._logger.debug("Vocab file: %s", vocab_file)

            #if not all([os.path.exists(path) for path in [engine_file_path, tokenizer_dir, vocab_file]]):
            #    raise FileNotFoundError("Required model components are missing.")
            use_py_session = kwargs['use_py_session'] if 'use_py_session' in kwargs else self._app_config_info['use_py_session']
            add_special_tokens = kwargs['add_special_tokens'] if 'add_special_tokens' in kwargs else self._app_config_info['add_special_tokens']
            trtLlm_debug_mode = kwargs['trtLlm_debug_mode'] if 'trtLlm_debug_mode' in kwargs else self._app_config_info['trtLlm_debug_mode']

            model_name, _ = read_model_name(model_path)
            prompt_template_obj = LLMPromptTemplate()
            text_qa_template_str = prompt_template_obj.model_context_template(model_name)

            self._llm = TrtLlmAPI(
                model_path=model_path,
                # engine_name="rank0.engine",
                tokenizer_dir=tokenizer_dir,
                temperature=model_info["metadata"].get("temperature", 0.1),
                max_new_tokens=model_info["metadata"].get("max_new_tokens", None),
                context_window=model_info["metadata"].get("max_input_token", None),
                vocab_file=vocab_file,
                completion_to_prompt=text_qa_template_str,
                use_py_session=use_py_session,
                add_special_tokens=add_special_tokens,
                trtLlm_debug_mode=trtLlm_debug_mode
            )
            return True
        except Exception as e:
            self._logger.error(f"Failed to init Llama-index TRTLLM model object: Error {str(e)}")
            return False

    def set_embedding_model(self, model_name, dim):
        """
        Set the embedding model for the language model.

        :param model_name: The name of the embedding model.
        :param dim: The dimension of the embedding model.
        """
        self._logger.debug("Setting embedding model with name: %s and dimension: %d", model_name, dim)
        try:
            self._embedding_model = HuggingFaceEmbedding(model_name=model_name)
            self._embedding_dim = dim
            self._logger.debug("Embedding model set successfully.")
        except Exception as e:
            self._logger.error("Failed to set embedding model: Error %s", str(e), exc_info=True)

    def set_rag_setting(self, **kwargs): #, chunk_size=1024, chunk_overlap=20, num_output=1024, context_window=3900):
        """
        Set the RAG (Retrieval-Augmented Generation) settings for the language model.

        :param llm: The language model object.
        :param kwargs
        """
        try:
            if self._embedding_model == None and self._embedding_dim == None:
                self.set_embedding_model(self._app_config_info["embedded_model"], self._app_config_info["embedded_dimension"])
            if self._llm == None:
                self._logger.error("LM model is null. Please create object of llm by calling init_llamaIndex_llm function")
                raise Exception("LLM model is null. Please create object of llm by calling init_llamaIndex_llm function")

            Settings.llm = self._llm
            Settings.embed_model = self._embedding_model
            Settings.node_parser = SentenceSplitter(chunk_size=kwargs['chunk_size'] if 'chunk_size' in kwargs else 1024,
                                                    chunk_overlap=kwargs['chunk_overlap'] if 'chunk_overlap' in kwargs else 20)

            self._logger.debug("RAG settings set successfully with settings\n "
                               f"Context Window: {Settings.context_window}"
                               f"Number of Output: {Settings.num_output}"
                               f"Chunk Size: {Settings.chunk_size}"
                               f"Chunk overlap: {Settings.chunk_overlap}")
        except Exception as e:
            self._logger.error("Failed to set RAG settings: Error %s", str(e), exc_info=True)

    def generate_query_engine(self, folder_path: str, streaming: bool = False, force_rewrite=False):
        """
        Generate a query engine for the language model.

        :param folder_path: The path to the folder containing data.
        :param streaming: Whether to enable streaming mode. Default is False.
        :param force_rewrite: Whether to forcefully rewrite existing data. Default is False.
        :return: The query engine object.
        """
        try:
            persist_dir = f"{folder_path}_vector_embedding"
            if force_rewrite:
                if os.path.exists(persist_dir):
                    self._logger.info("Force rewrite enabled. Deleting existing directory for a fresh start.")
                    self.delete_persist_dir(persist_dir)

            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                self._logger.info("Using the persisted value from %s", persist_dir)
                vector_store = FaissVectorStore.from_persist_dir(persist_dir)
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store, persist_dir=persist_dir
                )
                index = load_index_from_storage(storage_context=storage_context)
            else:
                self._logger.info("Generating new values")
                torch.cuda.empty_cache()
                gc.collect()
                documents = self._load_documents(folder_path)

                # Initialize FAISS index and load documents
                faiss_index = faiss.IndexFlatL2(self._embedding_dim)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # Create and return the query engine
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
                index.storage_context.persist(persist_dir=persist_dir)

            query_engine = index.as_query_engine(streaming=streaming,
                                                 similarity_top_k=self._app_config_info["similarity_top_k"])
            self._logger.debug("Query engine generated successfully.")
            torch.cuda.empty_cache()
            gc.collect()
            return query_engine

        except Exception as e:
            self._logger.error("Failed to generate the llama-index query engine: Error %s", str(e), exc_info=True)
            raise Exception("Failed to generate the llama-index query engine")

    def _load_documents(self, folder_path):
        """
        Load documents from the specified folder path.

        :param folder_path: The path to the folder containing data.
        :return: A list of loaded documents.
        """
        try:
            if os.path.exists(folder_path) and os.listdir(folder_path):
                file_metadata = lambda x: {"filename": x}
                documents = SimpleDirectoryReader(folder_path, file_metadata=file_metadata,
                                                  recursive=True,
                                                  required_exts=[".pdf", ".doc", ".docx", ".txt", ".xml"]).load_data()
            else:
                self._logger.info("No files found in the directory. Initializing an empty index.")
                documents = []
        except Exception as e:
            self._logger.error("Error occurred while initializing index: %s", str(e))
            documents = []
        return documents

    def delete_persist_dir(self, persist_dir):
        """
        Delete the persistence directory.

        :param persist_dir: The path to the persistence directory.
        """
        if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
            try:
                shutil.rmtree(persist_dir)
                self._logger.info("Persist directory %s deleted successfully.", persist_dir)
            except Exception as e:
                self._logger.error("Error occurred while deleting directory: %s", str(e))
                raise Exception(f"Error occurred while deleting directory: {str(e)}")

    def generate_response(self, query, query_engine):
        """
        Generate a response for a given query using the provided query engine.

        :param query: The query string for which to generate a response.
        :param query_engine: The query engine object to use for generating the response.
        :return: The generated response.
        """
        self._logger.debug("Generating response for query: %s", query)
        try:
            response = query_engine.query(query)
            torch.cuda.empty_cache()
            gc.collect()
            self._logger.debug("Response generated successfully.")
            return response
        except Exception as e:
            self._logger.error("Failed to generate response: Error %s", str(e), exc_info=True)
            raise Exception(f"Failed to generate the response: {str(e)}")

    def generate_stream_response(self, query, query_engine):
        """
        Generate a streaming response for a given query using the provided query engine.

        :param query: The query string for which to generate a streaming response.
        :param query_engine: The query engine object to use for generating the streaming response.
        :yields: Tokens and source nodes from the streaming response.
        """
        try:
            response = query_engine.query(query)
            return response
        except Exception as e:
            self._logger.error("Failed to generate stream response: Error %s", str(e), exc_info=True)
            raise Exception(f"Failed to generate the stream response: {str(e)}")

    def unload_llm(self):
        """
        Unload the currently loaded language model, if any.

        :raises Exception: If unloading the model fails.
        """
        self._logger.debug("Unloading LLM model.")
        if self._llm is not None:
            try:
                # Unload the language model
                self._llm.unload_llm()
                self._llm = None
            except Exception as e:
                self._logger.error("Failed to unload the language model: Error %s", str(e), exc_info=True)
                raise Exception(f"Failed to unload the language model: {str(e)}")
        else:
            self._logger.warning("No LLM model loaded to unload.")

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
