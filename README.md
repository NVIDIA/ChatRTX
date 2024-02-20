# 🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙



<p align="center">
<img src="https://github.com/NVIDIA/trt-llm-rag-windows/blob/release/1.0/media/rag-demo.gif?raw=true"  align="center">
</p>

> [!IMPORTANT]  
> Use this branch only for TRT-LLM v0.7.1 on Windows. 

This repository showcases a Retrieval-augmented Generation (RAG) pipeline implemented using the [llama_index](https://github.com/run-llama/llama_index) library for Windows. The pipeline incorporates the LLaMa 2 13B model, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), and the [FAISS](https://github.com/facebookresearch/faiss) vector search library. For demonstration, the dataset consists of thirty recent articles sourced from [NVIDIA Geforce News](https://www.nvidia.com/en-us/geforce/news/).



### What is RAG? 🔍
Retrieval-augmented generation (RAG) for large language models (LLMs) seeks to enhance prediction accuracy by leveraging an external datastore during inference. This approach constructs a comprehensive prompt enriched with context, historical data, and recent or relevant knowledge.

## Getting Started

Ensure you have the pre-requisites in place:

1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) using the Windows using the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/windows/README.md#quick-start).

> [!TIP]
> We recommend installing TensorRT-LLM in a virtual environment or Conda env. 

```
pip install tensorrt_llm==0.7.1 --extra-index-url https://pypi.nvidia.com  --extra-index-url https://download.pytorch.org/whl/cu121
```

3. Ensure you have access to the Llama 2 [repository on Huggingface](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

4. In this project, the LLaMa 2 13B AWQ 4bit quantized model is employed for inference. Before using it, you'll need to compile a TensorRT Engine specific to your GPU. Please refer to the [instructions](#building-trt-engine).


<h3 id="setup"> Setup Steps </h3>

1. Clone this repository: 
```
git clone https://github.com/NVIDIA/trt-llm-rag-windows.git
```
2. Place the TensorRT engine for LLaMa 2 13B model in the model/ directory
- Build the TRT engine by following the instructions provided [here](#building-trt-engine).
3. Acquire the llama2-13b chat tokenizer files (tokenizer.json, tokenizer.model and tokenizer_config.json) [here](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main).
4. Download AWQ weights for building the TensorRT engine llama_tp1_rank0.npz [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.3). 
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

Follow these steps to build your TRT engine:

Download LLaMa 2 13B chat tokenizer from [https://huggingface.co/meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf). Only download chat config & tokenizer files (config.json, tokenizer.json, tokenizer.model and tokenizer_config.json)

Download LLaMa 2 13B AWQ int4 checkpoints **llama_tp1_rank0.npz** and **llama_tp1.json** from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.3)

Clone the [TensorRT LLM](https://github.com/NVIDIA/TensorRT-LLM/) repository's v0.7.1 branch:
```
git clone --branch v0.7.1 https://github.com/NVIDIA/TensorRT-LLM.git
```

Navigate to the TensorRT-LLM repo directory and run the following script:
```
python examples\llama\build.py --model_dir <path to llama13_chat model> --quant_ckpt_path <path to model.npz> --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha  --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --output_dir <TRT engine folder>

```

If HF tokenizer & config.json are downloaded in the model\hf directory and the model's weights in model\weights directory within the tensorrt-llm dir, the build command looks like this:

```
python examples\llama\build.py --model_dir model\hf --quant_ckpt_path model\weights\llama_tp1_rank0.npz --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha  --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --output_dir trt-engine
```

Here, the model engine is stored in the modelout directory. Move the contents of the trt-engine directory into the trt-llm-rag-windows repo's model directory. Engine generation should create three new files inside the trt-engine directory: config.json, llama_float16_tp1_rank0.engine file, and model.cache. 

## Adding your own data
- This app loads data from the dataset/ directory into the vector store. To add support for your own data, replace the files in the dataset/ directory with your own data. By default, the script uses llamaindex's SimpleDirectoryLoader which supports text files in several platforms such as .txt, PDF, and so on.


This project requires additional third-party open source software projects as specified in the documentation. Review the license terms of these open source projects before use.
