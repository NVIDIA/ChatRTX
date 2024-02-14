# 🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙



<p align="center">
<img src="https://github.com/NVIDIA/trt-llm-rag-windows/blob/release/1.0/media/rag-demo.gif?raw=true"  align="center">
</p>


This repository showcases a Retrieval-augmented Generation (RAG) pipeline implemented using the [llama_index](https://github.com/run-llama/llama_index) library for Windows. The pipeline incorporates the LLaMa 2 13B model, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), and the [FAISS](https://github.com/facebookresearch/faiss) vector search library. For demonstration, the dataset consists of thirty recent articles sourced from [NVIDIA Geforce News](https://www.nvidia.com/en-us/geforce/news/).


### What is RAG? 🔍
Retrieval-augmented generation (RAG) for large language models (LLMs) seeks to enhance prediction accuracy by leveraging an external datastore during inference. This approach constructs a comprehensive prompt enriched with context, historical data, and recent or relevant knowledge.

## Getting Started

Ensure you have the pre-requisites in place:

1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) wheel for Windows and follow Quick Start instructions
   * TensorRT-LLM v0.5.0 [Quick Start](https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/windows/README.md#quick-start)
     ```
     pip install tensorrt_llm==0.5.0.post1 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
     ```
   * TensorRT-LLM v0.7.0 [Quick Start](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.0/windows/README.md#quick-start)
     ```
     pip install tensorrt_llm==0.7.0 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
     ```
   * TensorRT-LLM v0.7.1 [Quick Start](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/windows/README.md#quick-start)
     ```
     pip install tensorrt_llm==0.7.1 --extra-index-url https://pypi.nvidia.com  --extra-index-url https://download.pytorch.org/whl/cu121
     ```

2. Ensure you have access to the Llama 2 [repository on Huggingface](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf). Acquire the llama tokenizer (*tokenizer.json*, *tokenizer.model* and *tokenizer_config.json*) [here](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main).

3. In this project, the LLaMa 2 13B AWQ 4bit quantized model is employed for inference. Before using it, you'll need to compile a TensorRT Engine specific to your GPU. Please refer to the [instructions](#building-trt-engine) to build and validate TRT engine.

4. Clone this repository: 
   ```
   git clone https://github.com/NVIDIA/trt-llm-rag-windows.git
   git checkout release/1.0
   ```

5. Install the necessary libraries: 
   ```
   pip install -r requirements.txt
   ```
6. Launch the application using the following command:

   ```
   python app.py --trt_engine_path <TRT Engine folder> --trt_engine_name <TRT Engine file>.engine --tokenizer_dir_path <tokernizer folder> --data_dir <Data folder>

   ```
   In our case, that will be:

   ```
   python app.py --trt_engine_path model/ --trt_engine_name llama_float16_tp1_rank0.engine --tokenizer_dir_path model/ --data_dir dataset/
   ```

   >Note:
   >On its first run, this example will persist/cache the data folder in vector library. Any modifications in the data folder won't take effect until the "storage-default" cache directory is removed from the application directory.


## Detailed Command References 

```
python app.py --trt_engine_path <TRT Engine folder> --trt_engine_name <TRT Engine file>.engine --tokenizer_dir_path <tokernizer folder> --data_dir <Data folder>

```

Arguments

| Name | Details |
| ------ | ------ |
| --trt_engine_path <> | Directory of TensorRT engine |
| --trt_engine_name <> | Engine file name (e.g. llama_float16_tp1_rank0.engine)       |
| --tokenizer_dir_path <> | HF downloaded model files for tokenizer e.g. [https://huggingface.co/meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| --data_dir <> | Directory with context data (pdf, txt etc.) e.g. ".\dataset" |


<h3 id="building-trt-engine">Building TRT Engine</h3>

1. Download LLaMA 2 13B chat tokenizer files (*tokenizer.json*, *tokenizer.model* and *tokenizer_config.json*) from [https://huggingface.co/meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
 

2. Download LLaMA2 int4 checkpoints and build instructions

   | TensorRT-LLM version | Download TRT-LLM source | Engine Build Instruction | Download checkpoints |
   |--|--|--|--|
   | v0.5.0 | [TensorRT-LLM v0.5.0](https://github.com/NVIDIA/TensorRT-LLM/archive/refs/heads/release/0.5.0.zip) | [Readme.md](https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/windows/examples/llama/README.md) | Download *model.pt* from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2) |
   | v0.7.0 | [TensorRT-LLM v0.7.0](https://github.com/NVIDIA/TensorRT-LLM/archive/refs/tags/v0.7.0.zip) | [Readme.md](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.0/windows/examples/llama/README.md) | Download *llama_tp1.json* and *llama_tp1_rank0.npz* from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.3)|
   | v0.7.1 | [TensorRT-LLM v0.7.1](https://github.com/NVIDIA/TensorRT-LLM/archive/refs/tags/v0.7.1.zip) | [Readme.md](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/windows/examples/llama/README.md) | Download *llama_tp1.json* and *llama_tp1_rank0.npz* from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.3)|


3. Navigate to the examples\llama directory and follow build command based on Readme.md instructions to generate engine.
   * For TRT-LLM v0.5.0 build command is below
    ```
    python build.py --model_dir <path to llama13_chat model> --quant_ckpt_path <path to model.pt> --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --enable_context_fmha --max_batch_size 1 --max_input_len 3500 --max_output_len 1024 --output_dir <TRT engine folder>
    ```

4. Navigate to the examples\llama directory and follow run command based on Readme.md instructions to validate the engine.
   * For TRT-LLM v0.5.0 run command is below
    ```
    python3 run.py --max_output_len=50  --tokenizer_dir <tokenizer-path>  --engine_dir=<engine-path>
    ```

## Adding your own data
- This app loads data from the dataset/ directory into the vector store. To add support for your own data, replace the files in the dataset/ directory with your own data. By default, the script uses llamaindex's SimpleDirectoryLoader which supports text files in several platforms such as .txt, PDF, and so on.

>Note:
>This project requires additional third-party open source software projects as specified in the documentation. Review the license terms of these open source projects before use.
