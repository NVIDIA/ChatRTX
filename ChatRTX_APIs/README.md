# ChatRTX APIs

ChatRTX APIs allow developers to seamlessly integrate their applications with the TensorRT-LLM powered inference engine and utilize the various AI models supported by ChatRTX. This integration enables developers to incorporate advanced AI inference and RAG features into their applications.
These APIs serve as the foundation for the ChatRTX application. Please refer [API Examples](#api-examples) for examples of using the APIs.

## Key Features

1. **NVIDIA NIM**: The ChatRTX APIs enable use of NVIDIA NIMs for LLMs on supported GPUs. 

2. **TensorRT-LLM Inference Backend**: The ChatRTX APIs enable the use of the TensorRT-LLM inference backend, allowing for efficient and optimized AI model performance.

3. **Download and Build TensorRT-LLM Checkpoints**: With these APIs, you can download TensorRT-LLM checkpoints from NGC (NVIDIA GPU Cloud), build the TRT-LLM engine, and provide the necessary infrastructure to run inference and RAG with various AI models.

4. **Streaming and Non-Streaming Inference APIs**: Supports both streaming and non-streaming inference APIs, offering flexibility depending on the application’s requirements.

5. **RAG with Llama Index**: Provides a TRT-LLM connector to use TRT-LLM as the inference backend for RAG. It also includes the basic RAG pipeline with Llama Index and TRT-LLM. With the Llama Index TRT-LLM connector, users are free to add high-level RAG features.

6. **Models supported**:  

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

## Setup

If you are using [ChatRTX installer](https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/), the below steps can be skipped as setup is done by installer. If not using installer, manually install the components using following steps:

1. Run NIM Setup from [here](https://docs.nvidia.com/nim/wsl2/latest/getting-started.html#use-the-nvidia-nim-wsl2-installer-recommended)

2. Install [Python 3.10.11](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)

3. Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467). You will be prompted to choose between an exe, which installs the MPI executable, and an msi, which installs the MPI SDK. Download and install both

4. Download 'dependencies.zip' from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/chatrtx_05_dependencies/files?version=1.3). Extract the zip and download below wheels into same 'dependencies' folder along with other wheels.

5. Skip this step if using RTX 5xxx series GPU. Download TensorRT-LLM wheel ('tensorrt_llm-0.9.0-cp310-cp310-win_amd64.whl') from [here](./dist).

6. Download the ChatRTX API SDK ('ChatRTX-0.5.0-py3-none-any.whl') from [here](./dist)

7. Start commandline in dependencies folder with downloaded wheels and run following commands.

8. Skip this step if using RTX 5xxx series GPU. For others run following commands:
    ```
    python -m pip install tensorrt-bindings==9.3.0.post12.dev1 tensorrt-libs==9.3.0.post12.dev1 --no-index --find-links .
    python -m pip install tensorrt==9.3.0.post12.dev1 --no-index --find-links .
    python -m pip install tensorrt_llm-0.9.0-cp310-cp310-win_amd64.whl --no-index --find-links .
    ```

9. Then run following commands:
    ```
    python -m pip install ngcsdk-3.57.0-py3-none-any.whl --no-index --find-links .
    python -m pip install ChatRTX-0.5.0-py3-none-any.whl --no-index --find-links .
    python -m pip install grpcio==1.67.1 --no-index --find-links .
    ```

10. Only if using RTX 5xxx series GPU run following commands:

    ```
    python -m pip install torch-2.6.0+cu128.nv-cp310-cp310-win_amd64.whl --no-index --find-links .
    python -m pip install torchvision-0.20.0a0+cu128.nv-cp310-cp310-win_amd64.whl --no-index --find-links .
    ```


**Note**: This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## API Examples
There are example files in the ./example directory that demonstrate how to use the APIs for different models and features:

1. nim_inference.py: Demonstrates how to set up and run NIMs like Llama 3.1 on Windows.

2. nim_rag.py: Demonstrates how to enable the RAG pipeline with NIM LLM connector using Llama Index framework.

3. inference.py: Demonstrates how to set up and run an inference pipeline for LLM models like Llama, Mistral, Gemma, and ChatGLM using TRTLLM natively on Windows.

4. inference_streaming.py: Shows how to use APIs to enable the streaming feature for inference.

5. rag.py: Demonstrates how to enable the RAG pipeline with the TRT-LLM connector using the Llama Index framework.

6. clip.py: Provides examples of how to use the CLIP model with the provided APIs.

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
2. NVIDIA NIMs consume several gigabytes of data. To reclaim this storage space follow [instructions to delete NIMs](https://docs.nvidia.com/nim/wsl2/latest/deleting-nims.html).
