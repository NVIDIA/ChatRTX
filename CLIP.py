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
from llama_index.schema import TextNode
from llama_index import (    
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext)
import gc
from llama_index.schema import QueryBundle
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

def run_model_on_images(images_dir, model, processor, text, top_matches_path, min_clip_score = 18):
    """
    Run the CLIP model on images in 'images_dir' and 'text', and copy the results that cross threshold of 'min_clip_score' and
    return their paths.
    """
    images = []
    im_paths = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            path = os.path.join(images_dir, filename)
            images.append(Image.open(path).convert("RGB"))
            im_paths.append(path)
    # Move model to the correct device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if len(images) == 0:
        return []
    # Process images and move inputs to the same device as the model
    with torch.no_grad():
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
        outputs = model(**inputs)

    # Image-text similarity score  
    logits_per_image = outputs.logits_per_image    
    squeezed_logits = logits_per_image.squeeze()
    values, indices = squeezed_logits.sort(descending=True)    

    # Ensure the directory for top matches exists
    os.makedirs(top_matches_path, exist_ok=True)

    ret_paths = []
    # Save top matched images
    matches_images_count = 0
    for i, (value, idx) in enumerate(zip(values,indices)):
        if matches_images_count>0 and value.item() < min_clip_score:
            # return atleast 1 image even if score is not very good
            break
        top_image = images[idx.item()]
        cur_im_path = im_paths[idx.item()]
        original_full_name = os.path.basename(cur_im_path)
        path = os.path.join(top_matches_path, f"top_match_{i + 1}_{str(original_full_name)}")
        top_image.save(path)
        ret_paths.append(path)
        matches_images_count += 1

    return ret_paths


class CLIPEmbeddingStorageEngine:
    def __init__(self, data_dir, model_path, clip_model, clip_processor):
        self.data_dir = data_dir
        self.persist_dir = f"{self.data_dir}_clip_vector_embedding"
        self.index = None
        self.nodes = None
        os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.clip_model = clip_model
        self.clip_processor = clip_processor
    def create_nodes(self, force_rewrite=False):

        nodes = []
        images_dir = self.data_dir
        if os.path.exists(self.persist_dir) and force_rewrite == False:
            return
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

    def initialize_index(self, force_rewrite=False):
        # Check if the persist directory exists and delete it if force_rewrite is true
        if force_rewrite and os.path.exists(self.persist_dir):
            print("Deleting existing directory for a fresh start.")
            self.delete_persist_dir()

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print("Using the persisted value form " + self.persist_dir)
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context=storage_context)

        else:
            self.index = VectorStoreIndex(self.nodes)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            torch.cuda.empty_cache()
            gc.collect()
        self.retriever = self.index.as_retriever(similarity_top_k=500)

    def delete_persist_dir(self):
        if os.path.exists(self.persist_dir) and os.path.isdir(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
            except Exception as e:
                print(f"Error occurred while deleting directory: {str(e)}")

    def query(self, input_text, top_matches_path, min_clip_score = 18):
        tokenizer = CLIPTokenizer.from_pretrained(self.model_path)
        text_inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(self.device)
        text_features = self.clip_model.get_text_features(**text_inputs)
        test_query_bundle = QueryBundle(input_text, embedding=text_features.tolist()[0])
        retrieval_results = self.retriever.retrieve(test_query_bundle)

        # Ensure the directory for top matches exists
        os.makedirs(top_matches_path, exist_ok=True)

        ret_paths = []
        # Save top matched images
        matches_images_count = 0
        for i, res in enumerate(retrieval_results):
            if matches_images_count>0 and (res.get_score()*100) < min_clip_score:
                # return atleast 1 image even if score is not very good
                break
            original_full_name = os.path.basename(res.metadata['path'])
            path = os.path.join(top_matches_path, f"top_match_{i + 1}_{str(original_full_name)}")
            shutil.copy(res.metadata['path'], path)
            ret_paths.append(path)
            matches_images_count += 1

        return ret_paths
