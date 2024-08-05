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

from PIL import Image
import os
import torch
import shutil
from llama_index.core.schema import TextNode
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext
)
import gc
from llama_index.core.schema import QueryBundle
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from ChatRTX.logger import ChatRTXLogger
import ctypes

class CLIPEmbeddingStorageEngine:
    def __init__(self, data_dir, model_path, clip_model, clip_processor):
        try:
            self.data_dir = data_dir
            self.persist_dir = f"{self.data_dir}_clip_vector_embedding"
            self.index = None
            self.nodes = None
            os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            self.clip_model = clip_model
            self.clip_processor = clip_processor
            self._logger = ChatRTXLogger.get_logger()
        except Exception as e:
            self._logger.error(f"Initialization error: {str(e)}")

    def create_nodes(self, force_rewrite=False):
        try:
            nodes = []
            images_dir = self.data_dir
            if os.path.exists(self.persist_dir) and not force_rewrite:
                return True
            with torch.no_grad():
                # iterate over the image metadata dictionary and extracts image embeddings for each image
                for root, _, filenames in os.walk(images_dir):
                    for filename in filenames:
                        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            continue
                        img_file_path = os.path.join(root, filename)
                        if os.path.isfile(img_file_path):
                            image = Image.open(img_file_path).convert("RGB")
                            image_input = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                            image_features = self.clip_model.get_image_features(**image_input)
                            node = TextNode(text="dummy_text", metadata={"path": img_file_path}, embedding=image_features.tolist()[0])
                            nodes.append(node)
            self.nodes = nodes
            return True
        except Exception as e:
            self._logger.error(f"Error in create_nodes: {str(e)}")
            return False

    def initialize_index(self, force_rewrite=False):
        try:
            # Check if the persist directory exists and delete it if force_rewrite is true
            if force_rewrite and os.path.exists(self.persist_dir):
                print("Deleting existing directory for a fresh start.")
                self.delete_persist_dir()

            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                print("Using the persisted value from " + self.persist_dir)
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self.index = load_index_from_storage(storage_context=storage_context)
            else:
                self.index = VectorStoreIndex(self.nodes)
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                torch.cuda.empty_cache()
                gc.collect()
            self.retriever = self.index.as_retriever(similarity_top_k=500)
            return True
        except Exception as e:
            self._logger.error(f"Error in initialize_index: {str(e)}")
            return False

    def delete_persist_dir(self):
        try:
            if os.path.exists(self.persist_dir) and os.path.isdir(self.persist_dir):
                shutil.rmtree(self.persist_dir)
            return True
        except Exception as e:
            self._logger.error(f"Error occurred while deleting directory: {str(e)}")
            return False
            

    def is_junction(self, path):
        FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
        INVALID_FILE_ATTRIBUTES = -1
        
        attrs = ctypes.windll.kernel32.GetFileAttributesW(path)
        if attrs == INVALID_FILE_ATTRIBUTES:
            raise OSError("Failed to get file attributes for path: " + path)
        
        return bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT)

    def query(self, input_text, top_matches_path, min_clip_score):
        try:
            tokenizer = CLIPTokenizer.from_pretrained(self.model_path)
            text_inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(self.device)
            text_features = self.clip_model.get_text_features(**text_inputs)
            test_query_bundle = QueryBundle(input_text, embedding=text_features.tolist()[0])
            retrieval_results = self.retriever.retrieve(test_query_bundle)

            # Ensure the directory for top matches exists
            os.makedirs(top_matches_path, exist_ok=True)
            if os.path.islink(top_matches_path) or self.is_junction(top_matches_path):
                return False

            ret_paths = []
            matches_images_count = 0
            # Save top matched images
            for i, res in enumerate(retrieval_results):
                if matches_images_count > 0 and (res.get_score() * 100) < min_clip_score:
                    # return at least 1 image even if score is not very good
                    break
                original_full_name = os.path.basename(res.metadata['path'])
                path = os.path.join(top_matches_path, f"top_match_{i + 1}_{str(original_full_name)}")
                shutil.copy(res.metadata['path'], path)
                ret_paths.append(path)
                matches_images_count += 1
            return ret_paths if ret_paths else False
        except Exception as e:
            self._logger.error(f"Error in query: {str(e)}")
            return False

class ClipInference:
    def __init__(self):
        try:
            self.CLIPModel = None
            self.clip_model = None
            self.clip_processor = None
            self.clip_engine = None
            self.model_path = None
            self._logger = ChatRTXLogger.get_logger()
        except Exception as e:
            raise (f"Initialization error: {str(e)}")

    def load_model(self, model_path):
        self.model_path = model_path
        try:
            self.clip_model = CLIPModel.from_pretrained(self.model_path).to('cuda')
            self.clip_processor = CLIPProcessor.from_pretrained(self.model_path)
            return True
        except Exception as e:
            self._logger.error(f"Failed to init CLIP model object: Error {str(e)}")
            return False

    def generate_clip_engine(self, image_data_dir, force_rewrite=True):
        try:
            if self.clip_model is not None and self.clip_processor is not None:
                self.clip_engine = CLIPEmbeddingStorageEngine(image_data_dir, self.model_path, self.clip_model, self.clip_processor)
                if not self.clip_engine.create_nodes(force_rewrite):
                    return False
                if not self.clip_engine.initialize_index(force_rewrite):
                    return False
                return True
            else:
                self._logger.error("Model is not loaded. Please call load_model() first.")
                return False
        except Exception as e:
            self._logger.error(f"Failed to generate clip engine: Error {str(e)}")
            return False

    def generate_clip_response(self, input_text, top_matches_path, min_clip_score=23):
        try:
            # Check if the directory exists, if not, create it
            if not os.path.exists(top_matches_path):
                os.makedirs(top_matches_path)

            if self.clip_engine is not None:
                result = self.clip_engine.query(input_text, top_matches_path, min_clip_score)
                return result if result else False
            else:
                self._logger.error("Clip engine is not initialized. Please call generate_clip_engine() first.")
                return False
        except Exception as e:
            self._logger.error(f"Failed to generate clip response: Error {str(e)}")
            return False
