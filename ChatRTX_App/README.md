# 🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙

### Installer

If you are using [ChatRTX installer](https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/), setup of the models selected during installation is done by the installer. 

### ChatRTX Backend APIs

[ChatRTX APIs](https://github.com/NVIDIA/ChatRTX/tree/release/0.4.0/ChatRTX_APIs) serve as the foundation for the ChatRTX application, providing backend inference and RAG capabilities. These APIs allow developers to seamlessly integrate their applications with the inference engine and utilize the various AI models supported by ChatRTX. This integration enables developers to incorporate advanced AI inference and RAG features into their applications

### ChatRTX UI Client

# Prerequisite
- Node and NPM installed

## Run ChatRTX - Development
- Set the python enviromnent path in file config.js at src\bridge_commands
- In one terminal run `npm run watch`
- In another run `npm run start-electron`

## Build ChatRTX
- Set the python enviromnent path in file config-packed.js at src\bridge_commands if needed.
- Run `npm run build-electron`. This creates dist and contains NVIDIA ChatRTX.exe at location dist\win-unpacked which launches ChatRTX app

Note: Above does not generate signed binaries, binaries needs to signed when app is distirbuted publicly.

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
