# 🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙



<p align="center">
<img src="https://github.com/NVIDIA/trt-llm-rag-windows/blob/release/1.0/media/rag-demo.gif?raw=true"  align="center">
</p>


This repository showcases a Retrieval-augmented Generation (RAG) pipeline implemented using the [llama_index](https://github.com/run-llama/llama_index) library for Windows. The pipeline incorporates the LLaMa 2 13B model, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), and the [FAISS](https://github.com/facebookresearch/faiss) vector search library. For demonstration, the dataset consists of thirty recent articles sourced from [NVIDIA Geforce News](https://www.nvidia.com/en-us/geforce/news/).


### What is RAG? 🔍
Retrieval-augmented generation (RAG) for large language models (LLMs) seeks to enhance prediction accuracy by leveraging an external datastore during inference. This approach constructs a comprehensive prompt enriched with context, historical data, and recent or relevant knowledge.

## Getting Started

> [!IMPORTANT]  
> This branch is valid only for TRT-LLM v0.5. For the instructions compatible with the latest TRT-LLM release (v0.7.1), please switch to this branch: [trt-llm-0.7.1](https://github.com/NVIDIA/trt-llm-rag-windows/blob/trt-llm-0.7.1)


Ensure you have the pre-requisites in place:

1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) for Windows using the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/windows/README.md).

2. Ensure you have access to the Llama 2 [repository on Huggingface](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

3. In this project, the LLaMa 2 13B AWQ 4bit quantized model is employed for inference. Before using it, you'll need to compile a TensorRT Engine specific to your GPU. If you're using the GeForce RTX 4090 (TensorRT 9.1.0.4 and TensorRT-LLM release 0.5.0), the compiled TRT Engine is available for download [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2). For other  NVIDIA GPUs or TensorRT versions, please refer to the [instructions](#building-trt-engine).


<h3 id="setup"> Setup Steps </h3>

1. Clone this repository: 
```
git clone https://github.com/NVIDIA/trt-llm-rag-windows.git
```
2. Place the TensorRT engine for LLaMa 2 13B model in the model/ directory
- For GeForce RTX 4090 users: Download the pre-built TRT engine [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2) and place it in the model/ directory.
- For other NVIDIA GPU users: Build the TRT engine by following the instructions provided [here](#building-trt-engine).
3. Acquire the llama tokenizer (tokenizer.json, tokenizer.model and tokenizer_config.json) [here](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main).
4. Download AWQ weights for building the TensorRT engine model.pt [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2). (For RTX 4090, use the pregenerated engine provided earlier.)
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

For RTX 4090 (TensorRT 9.1.0.4 & TensorRT-LLM 0.5.0), a prebuilt TRT engine is provided. For other RTX GPUs or TensorRT versions, follow these steps to build your TRT engine:

Download LLaMa 2 13B chat model from [https://huggingface.co/meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

Download LLaMa 2 13B AWQ int4 checkpoints **model.pt** from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2)

Clone the [TensorRT LLM](https://github.com/NVIDIA/TensorRT-LLM/) repository:
```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

Navigate to the examples\llama directory and run the following script:
```
python build.py --model_dir <path to llama13_chat model> --quant_ckpt_path <path to model.pt> --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --enable_context_fmha --max_batch_size 1 --max_input_len 3000 --max_output_len 1024 --output_dir <TRT engine folder>
```

## Adding your own data
- This app loads data from the dataset/ directory into the vector store. To add support for your own data, replace the files in the dataset/ directory with your own data. By default, the script uses llamaindex's SimpleDirectoryLoader which supports text files in several platforms such as .txt, PDF, and so on.


This project requires additional third-party open source software projects as specified in the documentation. Review the license terms of these open source projects before use.
