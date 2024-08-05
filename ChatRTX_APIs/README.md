# ChatRTX APIs

ChatRTX APIs allow developers to seamlessly integrate their applications with the TensorRT-LLM powered inference engine and utilize the various AI models supported by ChatRTX. This integration enables developers to incorporate advanced AI inference and RAG features into their applications.
THese APIs serve as the foundation for the ChatRTX application.

## Key Features

1. **TensorRT-LLM Inference Backend**: The ChatRTX APIs enable the use of the TensorRT-LLM inference backend, allowing for efficient and optimized AI model performance.

2. **Download and Build TensorRT-LLM Checkpoints**: With these APIs, you can download TensorRT-LLM checkpoints from NGC (NVIDIA GPU Cloud), build the TRT-LLM engine, and provide the necessary infrastructure to run inference and RAG with various AI models.

3. **Streaming and Non-Streaming Inference APIs**: Supports both streaming and non-streaming inference APIs, offering flexibility depending on the application’s requirements.

4. **RAG with Llama Index**: Provides a TRT-LLM connector to use TRT-LLM as the inference backend for RAG. It also includes the basic RAG pipeline with Llama Index and TRT-LLM. With the Llama Index TRT-LLM connector, users are free to add high-level RAG features.

5. **Models supported**: 
The AI models that are supported in this app:
- LLaMa 2 13B
- Mistral 7B
- ChatGLM3 6B
- Whisper Medium (for supporting voice input)
- CLIP (for images)

## Setup

1. Install Python 3.10.11

2. Download and install [Microsoft] (https://www.microsoft.com/en-us/download/details.aspx?id=57467) MPI. You will be prompted to choose between an exe, which installs the MPI executable, and an msi, which installs the MPI SDK. Download and install both

3. Install TensorRT-LLM wheel. Please take the wheel(tensorrt_llm-0.9.0-cp310-cp310-win_amd64.whl) form [here](https://github.com/NVIDIA/ChatRTX/tree/release/0.4.0/ChatRTX_APIs/dist). 

    ```
    cd wheel
    pip install tensorrt_llm-0.9.0-cp310-cp310-win_amd64.whl --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
    ```

4. Download and install 'ngcsdk-3.41.2-py3-none-any.whl' from [here](https://catalog.canary.ngc.nvidia.com/orgs/nvidia/teams/ngc-apps/resources/ngc_sdk/files?version=3.41.2) and install it using the command below. This enables us to downloads from NGC:
    ```
    pip install .\ngcsdk-3.41.2-py3-none-any.whl
    ```

5. Download and install the ChatRTX API SDK present in the dist directory

    ```
    pip install ChatRTX-0.4.0-py3-none-any.whl
    ```

**Note**: This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## API Examples
There are example files in the ./example directory that demonstrate how to use the APIs for different models and features:

1. inference.py: Demonstrates how to set up and run an inference pipeline for LLM models like Llama, Mistral, Gemma, and ChatGLM.

2. inference_streaming.py: Shows how to use APIs to enable the streaming feature for inference.

3. rag.py: Demonstrates how to enable the RAG pipeline with the TRT-LLM connector using the Llama Index framework.

4. clip.py: Provides examples of how to use the CLIP model with the provided APIs.

**Note**: This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Troubleshooting

1. After the installation if you face error 

    ```
    No module named 'tensorrt_bindings' or No module named 'tensorrt'
    ```

    Use the below commands to reinstall the tensortRT 

    ```
    python -m pip uninstall -y tensorrt
    python -m pip install --pre --extra-index-url https://pypi.nvidia.com/ tensorrt==9.3.0.post12.dev1 --no-cache-dir
    Check: python -c "import tensorrt_llm; print(tensorrt_llm._utils.trt_version())"
    ```

