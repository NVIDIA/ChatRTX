# 🚀 RAG on Windows using TensorRT-LLM, NVIDIA NIM and LlamaIndex 🦙


ChatRTX is a demo app that lets you personalize a GPT large language model (LLM) connected to your own content—docs, notes, photos. Leveraging retrieval-augmented generation (RAG), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), [NVIDIA NIM microservices](https://docs.nvidia.com/nim/index.html) and RTX acceleration, you can query a custom chatbot to quickly get contextually relevant answers. This app also lets you give query through your voice. As it all runs locally on your Windows RTX PC, you’ll get fast and secure results.
ChatRTX supports various file formats, including text, pdf, doc/docx, xml, png, jpg, bmp. Simply point the application at the folder containing your files and it'll load them into the library in a matter of seconds.

ChatRTX supports following AI models:

| Model                                      | Supported GPUs |
|--------------------------------------------|----------------|
| LlaMa 3.1 8B NIM                           | RTX 6000 Ada, RTX GPUs 4080, 4090, 5080, 5090          |
| RIVA Parakeet 0.6B NIM (for supporting voice input) | RTX 6000 Ada, RTX GPUs 4080, 4090, 5080, 5090  |
| CLIP (for images)                          | RTX 6000 Ada, RTX 3xxx, RTX 4xxx, RTX 5080, RTX 5090          |
| Whisper Medium (for supporting voice input)| RTX 6000 Ada, RTX 3xxx and RTX 4xxx series GPUs that have at least 8GB of GPU memory         |
| Mistral 7B                                 | RTX 6000 Ada, RTX 3xxx and RTX 4xxx series GPUs that have at least 8GB of GPU memory          |
| ChatGLM3 6B                                | RTX 6000 Ada, RTX 3xxx and RTX 4xxx series GPUs that have at least 8GB of GPU memory          |
| LLaMa 2 13B                                | RTX 6000 Ada, RTX 3xxx and RTX 4xxx series GPUs that have at least 16GB of GPU memory          |
| Gemma 7B                                   | RTX 6000 Ada, RTX 3xxx and RTX 4xxx series GPUs that have at least 16GB of GPU memory          |

The pipeline incorporates the above AI models, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), [LlamaIndex](https://www.llamaindex.ai/) and the [FAISS](https://github.com/facebookresearch/faiss) vector search library. In the sample application here, we have a dataset consisting of recent articles sourced from [NVIDIA Geforce News](https://www.nvidia.com/en-us/geforce/news/).


### What is RAG? 🔍
Retrieval-augmented generation (RAG) for large language models (LLMs) that seeks to enhance prediction accuracy by connecting the LLM to your data during inference. This approach constructs a comprehensive prompt enriched with context, historical data, and recent or relevant knowledge.

### Repository details
- ChatRTX_APIs: ChatRTX APIs allow developers to seamlessly integrate their applications with the TensorRT-LLM powered inference engine and utilize the various AI models supported by ChatRTX. This integration enables developers to incorporate advanced AI inference and RAG features into their applications.
These APIs serve as the foundation for the ChatRTX application. More details in ChatRTX_APIs [directory](./ChatRTX_APIs/README.md). 

- ChatRTX_App: ChatRTX_App is a demo application that is build on top of ChatRTX APIs using electron container. The UI is build in React with Material UI libraries. More details about how to build the UI is in ChatRTX_App [directory](./ChatRTX_App/README.md). 

## Getting Started

### Hardware requirement
- NVIDIA GeForce RTX 5090 or 5080 GPU or NVIDIA RTX 600 Ada or NVIDIA GeForce RTX 30 or 40 Series GPU with at least 8GB of VRAM
- Windows 11 23H2 or 24H2
- Driver 572.16 or later

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
