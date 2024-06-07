# 🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙

<p align="center">
<img src="https://gitlab-master.nvidia.com/winai/trt-llm-rag-windows/-/raw/main/media/rag-demo.gif"  align="center">
</p>

ChatRTX is a demo app that lets you personalize a GPT large language model (LLM) connected to your own content—docs, notes, photos. Leveraging retrieval-augmented generation (RAG), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), and RTX acceleration, you can query a custom chatbot to quickly get contextually relevant answers. This app also lets you give query through your voice and lets you retreive images matching your voice or text input. And because it all runs locally on your Windows RTX PC or workstation, you’ll get fast and secure results.
ChatRTX supports various file formats, including text, pdf, doc/docx, xml, png, jpg, bmp. Simply point the application at the folder containing your files and it'll load them into the library in a matter of seconds.

The AI models that are supported in this app:
- LLaMa 2 13B
- Mistral 7B
- ChatGLM3 6B
- Whisper Medium (for supporting voice input)
- CLIP (for images)

The pipeline incorporates the above AI models, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), [LlamaIndex](https://www.llamaindex.ai/) and the [FAISS](https://github.com/facebookresearch/faiss) vector search library. In the sample application here, we have a dataset consists of recent articles sourced from [NVIDIA GeForce News](https://www.nvidia.com/en-us/geforce/news/).


### What is RAG? 🔍
Retrieval-augmented generation (RAG) for large language models (LLMs) seeks to enhance prediction accuracy by connecting the LLM to your data during inference. This approach constructs a comprehensive prompt enriched with context, historical data, and recent or relevant knowledge.

## Getting Started

### Hardware requirement
- ChatRTX is currently built for RTX 3xxx and RTX 4xxx series GPUs that have at least 8GB of GPU memory.
- 50 GB of available hard disk space
- Windows 10/11
- Driver 535.11 or later

### Installer

If you are using [ChatRTX installer](https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/), setup of the models selected during installation is done by the installer. You can skip the installation steps below, launch the installed 'NVIDIA ChatRTX' desktop icon, and refer to the [Use additional model](#use-additional-model) section to add additional models.

### Install Prerequisites


1. Install [Python 3.10.11](https://www.python.org/downloads/windows/) or create a virtual environment. 
    
    - create your virtual environment (recommended)

    ```
    python3.10 -m venv ChatRTX
    ```

    - activate your environment

    ```
    ChatRTX\Scripts\activate
    ```

    You can also use conda to create your virtual environment (optional)

    - create conda environment 

    ```
    conda create -n chatrtx_env python=3.10
    ```

    - activate your conda environment

    ```
    conda activate chatrtx_env
    ```

2. Clone ChatRTX code repo into a local dir (%ChatRTX Folder%) using [Git for Windows](https://git-scm.com/download/win), and install necessary dependencies. This directory will be the root directory for this guide. 
    ```
    git clone https://github.com/NVIDIA/trt-llm-rag-windows.git
    cd trt-llm-rag-windows # root dir
    
    #install dependencies
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu121
    ```
3. Install TensorRT-LLM wheel. The wheel is already present in the wheel directory. 

    ```
    cd wheel
    pip install tensorrt_llm-0.9.0-cp310-cp310-win_amd64.whl --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
    ```

4. Download 'ngcsdk-3.41.2-py3-none-any.whl' from [here](https://catalog.canary.ngc.nvidia.com/orgs/nvidia/teams/ngc-apps/resources/ngc_sdk/files?version=3.41.2) and install it using the command below. This enables us to downloads from NGC:
    ```
    pip install .\ngcsdk-3.41.2-py3-none-any.whl
    ```

5. Microsoft MPI
Download and install Microsoft [MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467). You will be prompted to choose between an exe, which installs the MPI executable, and an msi, which installs the MPI SDK. Download and install both.

### Setup Mistral AWQ INT4 model
In this project, we use the AWQ int4 quantized models for the LLMs. Before using it, you'll need to build a TensorRT Engine specific to your GPU. Below we have the steps to build the engine.

1. Create a model directory for Mistral Models

    ```
    cd model
    mkdir mistral_model
    cd mistral_model

    #Create the relevant directories
    mkdir engine model_checkpoints tokenizer
    ```

2. Download tokenizer files in model/mistral_model/tokenizer directory

    ``` 
    cd model/mistral_model/tokenizer

    #Use curl to download the tokenizer files
    "C:\Windows\System32\curl.exe" -L -o config.json "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/llama/mistral-7b-int4-chat/1.2/files?redirect=true&path=mistral7b_hf_tokenizer/config.json"
    "C:\Windows\System32\curl.exe" -L -o tokenizer.json "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/llama/mistral-7b-int4-chat/1.2/files?redirect=true&path=mistral7b_hf_tokenizer/tokenizer.json"
    "C:\Windows\System32\curl.exe" -L -o tokenizer.model "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/llama/mistral-7b-int4-chat/1.2/files?redirect=true&path=mistral7b_hf_tokenizer/tokenizer.model"
    "C:\Windows\System32\curl.exe" -L -o tokenizer_config.json "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/llama/mistral-7b-int4-chat/1.2/files?redirect=true&path=mistral7b_hf_tokenizer/tokenizer_config.json"

    ```

3. Download Mistral awq int4 engine checkpoints in model/mistral_model/model_checkpoints folder

    ```
    cd model/mistral_model/model_checkpoints

    #Use curl to download the model checkpoint files files
    "C:\Windows\System32\curl.exe" -L -o config.json "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/llama/mistral-7b-int4-chat/1.2/files?redirect=true&path=config.json"
	"C:\Windows\System32\curl.exe" -L -o license.txt "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/llama/mistral-7b-int4-chat/1.2/files?redirect=true&path=license.txt"
	"C:\Windows\System32\curl.exe" -L -o rank0.safetensors "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/llama/mistral-7b-int4-chat/1.2/files?redirect=true&path=rank0.safetensors"
	"C:\Windows\System32\curl.exe" -L -o README.txt "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/llama/mistral-7b-int4-chat/1.2/files?redirect=true&path=README.txt"
   
    ```

3. Build the Mistral TRT-LLM int4 AWQ Engine 

    ``` 
    #inside the root directory
    trtllm-build --checkpoint_dir .\model\mistral_model\model_checkpoints --output_dir .\model\mistral_model\engine --gpt_attention_plugin float16 --gemm_plugin float16 --max_batch_size 1 --max_input_len 7168 --max_output_len 1024 --context_fmha=enable --paged_kv_cache=disable --remove_input_padding=disable
    ```

    We use the following directories that we previously created for the build command:
    | Name | Details |
    | ------ | ------ |
    | --checkpoint_dir | TRT-LLM checkpoints directory |
    | --output_dir | TRT-LLM engine directory | 


    Refer to the [TRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM) to learn more about the various commands and parameters.

### Setup Whisper medium INT8 model

1. Create the directories to store the Whisper model

    ```
    cd model
    mkdir whisper
    cd whisper

    #Create the relevant directories
    mkdir whisper_assets whisper_medium_int8_engine
    ```

2. Download model weights and tokenizer

    ``` 
    cd model/whisper/whisper_assets

    #Use curl to download the tokenizer and model weights files 
    "C:\Windows\System32\curl.exe" -L -o mel_filters.npz "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
    "C:\Windows\System32\curl.exe" -L -o multilingual.tiktoken "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken"
    "C:\Windows\System32\curl.exe" -L -o medium.pt "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
    ```

3. Build command

    ``` 
    # call command form root_dir
    python .\whisper\build_files\build.py --output_dir .\model\whisper\whisper_medium_int8_engine --use_gpt_attention_plugin --use_gemm_plugin --use_bert_attention_plugin --enable_context_fmha --max_batch_size 1 --max_beam_width 1 --model_name medium --use_weight_only --model_dir .\model\whisper\whisper_assets
    ```

    We use the following directories that we previously created for the build command:
    | Name | Details |
    | ------ | ------ |
    | --checkpoint_dir | TRT-LLM checkpoints directory |
    | --output_dir | TRT-LLM engine directory | 


    Refer to the [TRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM) to learn more about the various commands and parameters.

### Get Embedding Model: 

1. Make the below directory structure in model folder 

    ``` 
    cd model
    mkdir multilingual-e5-base
    ```

2. Download the below 'multilingual-e5-base' embedding model file from [here](https://huggingface.co/intfloat/multilingual-e5-base/tree/d13f1b27baf31030b7fd040960d60d909913633f) 

    files to download: 1_Pooling/config.json, commit.txt, config.json, model.safetensors, modules.json, README.md, sentence_bert_config.json, sentencepiece.bpe.model, special_tokens_map.json, tokenizer.json, tokenizer_config.json


Building above two models are sufficient to run the app. Other models can be downloaded and built after running the app.

## Deploying the App

- ### Run App
Running below commands would launch the UI of app in your browser
    
    ``` 
    # call command form root_dir

    python verify_install.py

    python app.py

    ``` 
You can refer to [User Guide](https://nvidia.custhelp.com/app/answers/detail/a_id/5542/~/nvidia-chatrtx-user-guide) for additional information on using the app.

- ### Use additional model
1. In the app UI that gets launched in browser after running app.py, click on 'Add new models' in the 'AI model' section.
2. Select the model from drop down list, read the model license and check the box of 'License'
3. Click on 'Download models' icon to start the download of model files in the background.
4. After downloading finishes, click on the newly appearing button 'Install'. This will build the TRT LLM engine files if necessary.
5. The installed model will now show up in the 'Select AI model' drop down list.

- ### Deleting model
In case any model is not needed, model can be removed by:
1. Clicking on the gear icon on the top right of the UI.
2. Clicking on 'Delete AI model' icon adjacent to the model name.

## Using your own data
- By default this app loads data from the dataset/ directory into the vector store. To use your own data select the folder in the 'Dataset' section of UI.

## Known Issues and Limitations

The following known issues exist in the current version:
- The app currently works with Microsoft Edge and Google Chrome browsers.  Due to a bug, the application does not work with Firefox browser.
- The app does not remember context. This means follow up questions  will not be answered based on the context of the previous questions. For example, if you previously asked “What is the price of the RTX 4080 Super?” and follow that up with “What are its hardware specifications?”, the app will not know that you are asking about the RTX 4080 Super.
- The source file attribution in the response is not always correct.
- Unlikely case where the app gets stuck in an unusable state that cannot be resolved by restarting, could often be fixed by deleting the preferences.json file (by default located at C:\Users\<user>\AppData\Local\NVIDIA\ChatRTX\RAG\trt-llm-rag-windows-main\config\preferences.json) and restarting.



This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
