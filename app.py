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
import os
import sys
import time
import calendar
import json
from model_setup_manager import download_model_by_name, build_engine_by_name
import logging
import gc
import torch
from pathlib import Path
from trt_llama_api import TrtLlmAPI
from whisper.trt_whisper import WhisperTRTLLM, decode_audio_file
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from collections import defaultdict
from llama_index import ServiceContext
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import set_global_service_context
from faiss_vector_storage import FaissEmbeddingStorage
from ui.user_interface import MainInterface
from scipy.io import wavfile
import scipy.signal as sps
import numpy as np
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from CLIP import run_model_on_images, CLIPEmbeddingStorageEngine
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import shutil
from llm_prompt_templates import LLMPromptTemplate
from utils import (read_model_name)
import win32api
import win32security

selected_CLIP = False
clip_engine = None
selected_ChatGLM = False
app_config_file = 'config\\app_config.json'
model_config_file = 'config\\config.json'
preference_config_file = 'config\\preferences.json'
data_source = 'directory'

# Use GetCurrentProcess to get a handle to the current process
hproc = win32api.GetCurrentProcess()
# Use GetCurrentProcessToken to get the token of the current process
htok = win32security.OpenProcessToken(hproc, win32security.TOKEN_QUERY)

# Retrieve the list of privileges enabled
privileges = win32security.GetTokenInformation(htok, win32security.TokenPrivileges)

# Iterate over privileges and output the ones that are enabled
priv_list = []
for priv_id, priv_flags in privileges:
    # Check if privilege is enabled
    if priv_flags == win32security.SE_PRIVILEGE_ENABLED or win32security.SE_PRIVILEGE_ENABLED_BY_DEFAULT:
        # Lookup the name of the privilege
        priv_name = win32security.LookupPrivilegeName(None, priv_id)
        priv_list.append(priv_name)

print(f"Privileges of app process: {priv_list}")

def read_config(file_name):
    try:
        with open(file_name, 'r', encoding='utf8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except json.JSONDecodeError:
        print(f"There was an error decoding the JSON from the file {file_name}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def get_model_config(config, model_name=None):
        selected_model = next((model for model in config["models"]["supported"] if model["name"] == model_name),
                              config["models"]["supported"][0])
        metadata = selected_model["metadata"]

        cwd = os.getcwd()  # Current working directory, to avoid calling os.getcwd() multiple times

        if "ngc_model_name" in selected_model:
            return {
                "model_path": os.path.join(cwd, "model", selected_model["id"], "engine") if "id" in selected_model else None,
                "engine": metadata.get("engine", None),
                "tokenizer_path": os.path.join(cwd, "model", selected_model["id"] ,selected_model["prerequisite"]["tokenizer_local_dir"] ) if "tokenizer_local_dir" in selected_model["prerequisite"] else None,
                "vocab": os.path.join(cwd, "model", selected_model["id"] ,selected_model["prerequisite"]["vocab_local_dir"], selected_model["prerequisite"]["tokenizer_files"]["vocab_file"]) if "vocab_local_dir" in selected_model["prerequisite"] else None,
                "max_new_tokens": metadata.get("max_new_tokens", None),
                "max_input_token": metadata.get("max_input_token", None),
                "temperature": metadata.get("temperature", None),
                "prompt_template": metadata.get("prompt_template", None)
            }
        elif "hf_model_name" in selected_model:
            return {
                "model_path": os.path.join(cwd, "model", selected_model["id"]) if "id" in selected_model else None,
                "tokenizer_path": os.path.join(cwd, "model", selected_model["id"]) if "id" in selected_model else None,
                "prompt_template": metadata.get("prompt_template", None)
            }

		
def get_asr_model_config(config, model_name=None):
    models = config["models"]["supported_asr"]
    selected_model = next((model for model in models if model["name"] == model_name), models[0])
    return {
        "model_path": os.path.join(os.getcwd(), selected_model["metadata"]["model_path"]),
        "assets_path": os.path.join(os.getcwd(), selected_model["metadata"]["assets_path"])
    }

def get_data_path(config):
    return os.path.join(os.getcwd(), config["dataset"]["path"])

# read the app specific config
app_config = read_config(app_config_file)
streaming = app_config["streaming"]
similarity_top_k = app_config["similarity_top_k"]
is_chat_engine = app_config["is_chat_engine"]
embedded_model_name = app_config["embedded_model"]
embedded_model = os.path.join(os.getcwd(), "model", embedded_model_name)
embedded_dimension = app_config["embedded_dimension"]
use_py_session = app_config["use_py_session"]
trtLlm_debug_mode = app_config["trtLlm_debug_mode"]
add_special_tokens = app_config["add_special_tokens"]
verbose = app_config["verbose"]

# read model specific config
selected_model_name = None
selected_data_directory = None
config = read_config(model_config_file)
if os.path.exists(preference_config_file):
    perf_config = read_config(preference_config_file)
    selected_model_name = perf_config.get('models', {}).get('selected')
    selected_data_directory = perf_config.get('dataset', {}).get('path')

if selected_model_name == None:
    selected_model_name = config["models"].get("selected")

if selected_model_name == "CLIP":
    selected_CLIP = True
if selected_model_name == "ChatGLM 3 6B int4 (Supports Chinese)":
    selected_ChatGLM = True

model_config = get_model_config(config, selected_model_name)
data_dir = config["dataset"]["path"] if selected_data_directory == None else selected_data_directory

asr_model_name = "Whisper Medium Int8"
asr_model_config = get_asr_model_config(config, asr_model_name)
asr_engine_path = asr_model_config["model_path"]
asr_assets_path = asr_model_config["assets_path"]

whisper_model = None
whisper_model_loaded = False
enable_asr = config["models"]["enable_asr"]
nvmlInit()

def generate_inferance_engine(data, force_rewrite=False):
    """
       Initialize and return a FAISS-based inference engine.

       Args:
           data: The directory where the data for the inference engine is located.
           force_rewrite (bool): If True, force rewriting the index.

       Returns:
           The initialized inference engine.

       Raises:
           RuntimeError: If unable to generate the inference engine.
       """
    try:
        global engine
        faiss_storage = FaissEmbeddingStorage(data_dir=data,
                                              dimension=embedded_dimension)
        faiss_storage.initialize_index(force_rewrite=force_rewrite)
        engine = faiss_storage.get_engine(is_chat_engine=is_chat_engine, streaming=streaming,
                                          similarity_top_k=similarity_top_k)
    except Exception as e:
        raise RuntimeError(f"Unable to generate the inference engine: {e}")

def generate_clip_engine(data_dir, model_path, clip_model, clip_processor, force_rewrite=False):
    global clip_engine
    clip_engine = CLIPEmbeddingStorageEngine(data_dir, model_path, clip_model, clip_processor)
    clip_engine.create_nodes(force_rewrite)
    clip_engine.initialize_index(force_rewrite)

llm = None
embed_model = None
service_context = None
clip_model = None
clip_processor = None

if selected_CLIP:
# Initialize model and processor
    clip_model = CLIPModel.from_pretrained(model_config["model_path"]).to('cuda')
    clip_processor = CLIPProcessor.from_pretrained(model_config["model_path"])
    generate_clip_engine(data_dir, model_config["model_path"], clip_model, clip_processor)
else:
    # create trt_llm engine object
    model_name, _ = read_model_name(model_config["model_path"])
    prompt_template_obj = LLMPromptTemplate()
    text_qa_template_str = prompt_template_obj.model_context_template(model_name)
    selected_completion_to_prompt =  text_qa_template_str
    llm = TrtLlmAPI(
        model_path=model_config["model_path"],
        engine_name=model_config["engine"],
        tokenizer_dir=model_config["tokenizer_path"],
        temperature=model_config["temperature"],
        max_new_tokens=model_config["max_new_tokens"],
        context_window=model_config["max_input_token"],
        vocab_file=model_config["vocab"],
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=selected_completion_to_prompt,
        use_py_session=use_py_session,
        add_special_tokens=add_special_tokens,
        trtLlm_debug_mode=trtLlm_debug_mode,
        verbose=verbose
    )

    # create embeddings model object
    embed_model = HuggingFaceEmbeddings(model_name=embedded_model)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model,
                                                context_window=model_config["max_input_token"], chunk_size=512,
                                                chunk_overlap=200)
    set_global_service_context(service_context)

    # load the vectorstore index
    generate_inferance_engine(data_dir)

def call_llm_streamed(query):
    partial_response = ""
    response = llm.stream_complete(query, formatted=False)
    for token in response:
        partial_response += token.delta
        yield partial_response

def chatbot(query, chat_history, session_id):
    if selected_CLIP:
        ts = calendar.timegm(time.gmtime())
        temp_image_folder_name = "Temp/Temp_Images"
        if os.path.isdir(temp_image_folder_name):                
            try:
                shutil.rmtree(os.path.join(os.getcwd(), temp_image_folder_name))
            except Exception as e:
                print("Exception during folder delete", e)
        image_results_path = os.path.join(os.getcwd(), temp_image_folder_name, str(ts))
        res_im_paths = clip_engine.query(query, image_results_path)
        if len(res_im_paths) == 0:
            yield "No supported images found in the selected folder"
            torch.cuda.empty_cache()
            gc.collect()
            return

        div_start = '<div class="chat-output-images">'
        div_end = '</div>'
        im_elements = ''
        for i, im in enumerate(res_im_paths):
            if i>2 : break # display atmost 3 images.
            cur_data_link_src = temp_image_folder_name +"/" + str(ts) + "/" + os.path.basename(im)
            cur_src = "file/" + temp_image_folder_name +"/" + str(ts) + "/" + os.path.basename(im)
            im_elements += '<img data-link="{data_link_src}" src="{src}"/>'.format(src=cur_src, data_link_src=cur_data_link_src)
        full_div = (div_start + im_elements + div_end)
        folder_link = f'<a data-link="{image_results_path}">{"See all matches"}</a>'
        prefix = ""
        if(len(res_im_paths)>1): 
            prefix = "Here are the top matching pictures from your dataset"
        else:
            prefix = "Here is the top matching picture from your dataset"
        response = prefix + "<br>"+ full_div + "<br>"+ folder_link

        gc.collect()
        torch.cuda.empty_cache()
        yield response
        torch.cuda.empty_cache()
        gc.collect()
        return

    if data_source == "nodataset":
        yield llm.complete(query, formatted=False).text
        return

    if is_chat_engine:
        response = engine.chat(query)
    else:
        response = engine.query(query)

    lowest_score_file = None
    lowest_score = sys.float_info.max
    for node in response.source_nodes:
        metadata = node.metadata
        if 'filename' in metadata:
            if node.score < lowest_score:
                lowest_score = node.score
                lowest_score_file = metadata['filename']

    file_links = []
    seen_files = set()  # Set to track unique file names
    ts = calendar.timegm(time.gmtime())
    temp_docs_folder_name = "Temp/Temp_Docs"
    docs_path = os.path.join(os.getcwd(), temp_docs_folder_name, str(ts))
    os.makedirs(docs_path, exist_ok=True)

    # Generate links for the file with the highest aggregated score
    if lowest_score_file:
        abs_path = Path(os.path.join(os.getcwd(), lowest_score_file.replace('\\', '/')))
        file_name = os.path.basename(abs_path)
        doc_path = os.path.join(docs_path, file_name)
        shutil.copy(abs_path, doc_path)

        if file_name not in seen_files:  # Ensure the file hasn't already been processed
            if data_source == 'directory':
                file_link = f'<a data-link="{doc_path}">{file_name}</a>'
            else:
                exit("Wrong data_source type")
            file_links.append(file_link)
            seen_files.add(file_name)  # Mark file as processed

    response_txt = str(response)
    if file_links:
        response_txt += "<br>Reference files:<br>" + "<br>".join(file_links)
    if not lowest_score_file:  # If no file with a high score was found
        response_txt = llm.complete(query).text
    yield response_txt

def stream_chatbot(query, chat_history, session_id):
    
    if selected_CLIP:
        ts = calendar.timegm(time.gmtime())
        temp_image_folder_name = "Temp/Temp_Images"
        if os.path.isdir(temp_image_folder_name):                
            try:
                shutil.rmtree(os.path.join(os.getcwd(), temp_image_folder_name))
            except Exception as e:
                print("Exception during folder delete", e)
        image_results_path = os.path.join(os.getcwd(), temp_image_folder_name, str(ts))
        res_im_paths = clip_engine.query(query, image_results_path)
        if len(res_im_paths) == 0:
            yield "No supported images found in the selected folder"
            torch.cuda.empty_cache()
            gc.collect()
            return
        div_start = '<div class="chat-output-images">'
        div_end = '</div>'
        im_elements = ''
        for i, im in enumerate(res_im_paths):
            if i>2 : break # display atmost 3 images.
            cur_data_link_src = temp_image_folder_name +"/" + str(ts) + "/" + os.path.basename(im)
            cur_src = "file/" + temp_image_folder_name +"/" + str(ts) + "/" + os.path.basename(im)
            im_elements += '<img data-link="{data_link_src}" src="{src}"/>'.format(src=cur_src, data_link_src=cur_data_link_src)
        full_div = (div_start + im_elements + div_end)
        folder_link = f'<a data-link="{image_results_path}">{"See all matches"}</a>'
        prefix = ""
        if(len(res_im_paths)>1): 
            prefix = "Here are the top matching pictures from your dataset"
        else:
            prefix = "Here is the top matching picture from your dataset"
        response = prefix + "<br>"+ full_div + "<br>"+ folder_link
        yield response
        torch.cuda.empty_cache()
        gc.collect()
        return

    if data_source == "nodataset":
        for response in call_llm_streamed(query):
            yield response
        return

    if is_chat_engine:
        response = engine.stream_chat(query)
    else:
        response = engine.query(query)

    partial_response = ""
    if len(response.source_nodes) == 0:
        response = llm.stream_complete(query, formatted=False)
        for token in response:
            partial_response += token.delta
            yield partial_response
    else:
        # Aggregate scores by file
        lowest_score_file = None
        lowest_score = sys.float_info.max

        for node in response.source_nodes:
            if 'filename' in node.metadata:
                if node.score < lowest_score:
                    lowest_score = node.score
                    lowest_score_file = node.metadata['filename']

        file_links = []
        seen_files = set()
        for token in response.response_gen:
            partial_response += token
            yield partial_response
            time.sleep(0.05)

        time.sleep(0.2)
        ts = calendar.timegm(time.gmtime())
        temp_docs_folder_name = "Temp/Temp_Docs"
        docs_path = os.path.join(os.getcwd(), temp_docs_folder_name, str(ts))
        os.makedirs(docs_path, exist_ok=True)

        if lowest_score_file:
            abs_path = Path(os.path.join(os.getcwd(), lowest_score_file.replace('\\', '/')))
            file_name = os.path.basename(abs_path)
            doc_path = os.path.join(docs_path, file_name)
            shutil.copy(abs_path, doc_path)
            if file_name not in seen_files:  # Check if file_name is already seen
                if data_source == 'directory':
                    file_link = f'<a data-link="{doc_path}">{file_name}</a>'
                else:
                    exit("Wrong data_source type")
                file_links.append(file_link)
                seen_files.add(file_name)  # Add file_name to the set

        if file_links:
            partial_response += "<br>Reference files:<br>" + "<br>".join(file_links)
        yield partial_response

    # call garbage collector after inference
    torch.cuda.empty_cache()
    gc.collect()


interface = MainInterface(chatbot=stream_chatbot if streaming else chatbot, streaming=streaming)


def on_shutdown_handler(session_id):
    global llm, whisper_model, clip_model, clip_processor, clip_engine
    import gc
    if whisper_model is not None:        
        whisper_model.unload_model()
        del whisper_model
        whisper_model = None
    if llm is not None:
        llm.unload_model()
        del llm
        llm = None
    if clip_model is not None:
        del clip_model
        del clip_processor
        del clip_engine
        clip_model = None
        clip_processor = None
        clip_engine = None
    temp_data_folder_name = "Temp"
    if os.path.isdir(temp_data_folder_name):                
        try:
            shutil.rmtree(os.path.join(os.getcwd(), temp_data_folder_name))
        except Exception as e:
            print("Exception during temp folder delete", e)
    # Force a garbage collection cycle
    gc.collect()


interface.on_shutdown(on_shutdown_handler)


def reset_chat_handler(session_id):
    global faiss_storage
    global engine
    print('reset chat called', session_id)
    if selected_CLIP:
        return
    if is_chat_engine == True:
        faiss_storage.reset_engine(engine)


interface.on_reset_chat(reset_chat_handler)


def on_dataset_path_updated_handler(source, new_directory, video_count, session_id):
    print('data set path updated to ', source, new_directory, video_count, session_id)
    global engine
    global data_dir
    if selected_CLIP:
        data_dir = new_directory
        generate_clip_engine(data_dir, model_config["model_path"], clip_model, clip_processor)
        return
    if source == 'directory':
        if data_dir != new_directory:
            data_dir = new_directory
            generate_inferance_engine(data_dir)


interface.on_dataset_path_updated(on_dataset_path_updated_handler)


def on_model_change_handler(model, model_info, session_id):
    global llm, embedded_model, engine, data_dir, service_context, clip_model, clip_processor, selected_CLIP, selected_model_name, embed_model, model_config, selected_ChatGLM, clip_engine
    selected_model_name = model
    selected_ChatGLM = False
    
    if llm is not None:
        llm.unload_model()
        del llm
        llm = None
    
    if clip_model != None:
        del clip_model
        clip_model = None
        del clip_processor
        clip_processor = None
        del clip_engine
        clip_engine = None

    torch.cuda.empty_cache()
    gc.collect()
    
    cwd = os.getcwd()
    model_config = get_model_config(config, selected_model_name)

    selected_CLIP = False
    if selected_model_name == "CLIP":
        selected_CLIP = True
        if clip_model == None:
            clip_model = CLIPModel.from_pretrained(model_config["model_path"]).to('cuda')
            clip_processor = CLIPProcessor.from_pretrained(model_config["model_path"])
        generate_clip_engine(data_dir, model_config["model_path"], clip_model, clip_processor)
        return

    model_path = os.path.join(cwd, "model", model_info["id"], "engine") if "id" in model_info else None
    engine_name = model_info["metadata"].get('engine', None)

    if not model_path or not engine_name:
        print("Model path or engine not provided in metadata")
        return

    if selected_model_name == "ChatGLM 3 6B int4 (Supports Chinese)":
        selected_ChatGLM = True

    model_name, _ = read_model_name(model_path)
    prompt_template = LLMPromptTemplate()
    text_qa_template_str = prompt_template.model_context_template(model_name)
    selected_completion_to_prompt =  text_qa_template_str

    #selected_completion_to_prompt = chatglm_completion_to_prompt if selected_ChatGLM else completion_to_prompt
    llm = TrtLlmAPI(
        model_path=model_path,
        engine_name=engine_name,
        tokenizer_dir=os.path.join(cwd, "model", model_info["id"] ,model_info["prerequisite"]["tokenizer_local_dir"] ) if "tokenizer_local_dir" in model_info["prerequisite"] else None,
        temperature=model_info["metadata"].get("temperature"),
        max_new_tokens=model_info["metadata"].get("max_new_tokens"),
        context_window=model_info["metadata"].get("max_input_token"),
        vocab_file=os.path.join(cwd, "model", model_info["id"] ,model_info["prerequisite"]["vocab_local_dir"], model_info["prerequisite"]["tokenizer_files"]["vocab_file"]) if "vocab_local_dir" in model_info["prerequisite"] else None,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=selected_completion_to_prompt,
        use_py_session=use_py_session,
        add_special_tokens=add_special_tokens,
        trtLlm_debug_mode=trtLlm_debug_mode,
        verbose=verbose
    )
    if embed_model is None : embed_model = HuggingFaceEmbeddings(model_name=embedded_model)
    if service_context is None:
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model,
                                                    context_window=model_config["max_input_token"], chunk_size=512,
                                                    chunk_overlap=200)
    else:
        service_context = ServiceContext.from_service_context(service_context=service_context, llm=llm)
    set_global_service_context(service_context)
    generate_inferance_engine(data_dir)


interface.on_model_change(on_model_change_handler)


def on_dataset_source_change_handler(source, path, session_id):

    global data_source, data_dir, engine
    data_source = source

    if data_source == "nodataset":
        print(' No dataset source selected', session_id)
        return
    
    print('dataset source updated ', source, path, session_id)
    
    if data_source == "directory":
        data_dir = path
    else:
        print("Wrong data type selected")
    generate_inferance_engine(data_dir)


interface.on_dataset_source_updated(on_dataset_source_change_handler)

def handle_regenerate_index(source, path, session_id):
    if selected_CLIP:
        generate_clip_engine(data_dir, model_config["model_path"], clip_model, clip_processor, force_rewrite=True)
    else:
        generate_inferance_engine(path, force_rewrite=True)
    print("on regenerate index", source, path, session_id)


def mic_init_handler():
    global whisper_model, whisper_model_loaded, enable_asr
    enable_asr = config["models"]["enable_asr"]
    if not enable_asr:
        return False
    vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    free_vid_mem = vid_mem_info.free / (1024*1024)
    print("free video memory in MB = ", free_vid_mem)
    if whisper_model is not None:
        whisper_model.unload_model()
        del whisper_model
        whisper_model = None
    whisper_model = WhisperTRTLLM(asr_engine_path, assets_dir=asr_assets_path)
    whisper_model_loaded = True
    return True

interface.on_mic_button_click(mic_init_handler)

def mic_recording_done_handler(audio_path):
    transcription = ""
    global whisper_model, enable_asr, whisper_model_loaded
    if not enable_asr:
        return ""
    
    # Check and wait until model is loaded before running it.
    checks_left_for_model_loading = 40
    sleep_time = 0.2
    while checks_left_for_model_loading>0 and not whisper_model_loaded:
        time.sleep(sleep_time)
        checks_left_for_model_loading -= 1
    assert checks_left_for_model_loading>0, f"Whisper model loading not finished even after {(checks_left_for_model_loading*sleep_time)} seconds"
    if checks_left_for_model_loading == 0:
        return ""

    # Covert the audio file into required sampling rate
    current_sampling_rate, data = wavfile.read(audio_path)
    new_sampling_rate = 16000
    number_of_samples = round(len(data) * float(new_sampling_rate) / current_sampling_rate)
    data = sps.resample(data, number_of_samples)
    new_file_path = os.path.join( os.path.dirname(audio_path), "whisper_audio_input.wav" )
    wavfile.write(new_file_path, new_sampling_rate, data.astype(np.int16))
    language = "english"
    if selected_ChatGLM: language = "chinese"
    transcription = decode_audio_file( new_file_path, whisper_model, language=language, mel_filters_dir=asr_assets_path)

    if whisper_model is not None:        
        whisper_model.unload_model()
        del whisper_model
        whisper_model = None
    whisper_model_loaded = False
    return transcription

interface.on_mic_recording_done(mic_recording_done_handler)

def model_download_handler(model_info):
    download_path = os.path.join(os.getcwd(), "model")
    status = download_model_by_name(model_info=model_info,  download_path=download_path)
    print(f"Model download status: {status}")
    return status

interface.on_model_downloaded(model_download_handler)

def model_install_handler(model_info):
    download_path = os.path.join(os.getcwd(), "model")
    global llm, service_context
    #unload the current model
    if llm is not None:
        llm.unload_model()
        del llm
        llm = None
    # build the engine
    status = build_engine_by_name(model_info=model_info , download_path= download_path)
    print(f"Engine build status: {status}")
    return status

interface.on_model_installed(model_install_handler)

def model_delete_handler(model_info):
    print("Model deleting ", model_info)
    model_dir = os.path.join(os.getcwd(), "model", model_info['id'])
    isSuccess = True
    if os.path.isdir(model_dir): 
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print("Exception during temp folder delete", e)
            isSuccess = False
    return isSuccess

interface.on_model_delete(model_delete_handler)

interface.on_regenerate_index(handle_regenerate_index)
# render the interface
interface.render()