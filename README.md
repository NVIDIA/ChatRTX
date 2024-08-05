# 🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙

<p align="center">
<img src="https://gitlab-master.nvidia.com/winai/trt-llm-rag-windows/-/raw/main/media/rag-demo.gif"  align="center">
</p>

ChatRTX is a demo app that lets you personalize a GPT large language model (LLM) connected to your own content—docs, notes, photos. Leveraging retrieval-augmented generation (RAG), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), and RTX acceleration, you can query a custom chatbot to quickly get contextually relevant answers. This app also lets you give query through your voice. As it all runs locally on your Windows RTX PC, you’ll get fast and secure results.
ChatRTX supports various file formats, including text, pdf, doc/docx, xml, png, jpg, bmp. Simply point the application at the folder containing your files and it'll load them into the library in a matter of seconds.

The AI models that are supported in this app:
- LLaMa 2 13B
- Mistral 7B
- ChatGLM3 6B
- Whisper Medium (for supporting voice input)
- CLIP (for images)

The pipeline incorporates the above AI models, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), [LlamaIndex](https://www.llamaindex.ai/) and the [FAISS](https://github.com/facebookresearch/faiss) vector search library. In the sample application here, we have a dataset consisting of recent articles sourced from [NVIDIA Gefore News](https://www.nvidia.com/en-us/geforce/news/).


### What is RAG? 🔍
Retrieval-augmented generation (RAG) for large language models (LLMs) that seeks to enhance prediction accuracy by connecting the LLM to your data during inference. This approach constructs a comprehensive prompt enriched with context, historical data, and recent or relevant knowledge.

### Repository details
- ChatRTX_APIs: ChatRTX APIs allow developers to seamlessly integrate their applications with the TensorRT-LLM powered inference engine and utilize the various AI models supported by ChatRTX. This integration enables developers to incorporate advanced AI inference and RAG features into their applications.
These APIs serve as the foundation for the ChatRTX application. More details in ChatRTX_APIs [directory](https://github.com/NVIDIA/ChatRTX/blob/release/0.4.0/ChatRTX_APIs/README.md). 

- ChatRTX_App: ChatRTX_App is a demo application that is build on top of ChatRTX APIs using electron container. The UI is build in React with Material UI libraries. More details about how to build the UI is in ChatRTX_App [directory](https://github.com/NVIDIA/ChatRTX/blob/release/0.4.0/ChatRTX_App/README.md). 

## Getting Started

### Hardware requirement
- ChatRTX is currently built for RTX 3xxx and RTX 4xxx series GPUs that have at least 8GB of GPU memory.
- Windows 10/11
- Driver 535.11 or later

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
