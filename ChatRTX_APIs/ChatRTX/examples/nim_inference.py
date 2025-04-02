# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from openai import OpenAI
import os

from ChatRTX.model_manager.nim_manager import NIMManager

def initialize_nim_manager(nim_profile):
    nim_manager = NIMManager.get_instance()
    if not nim_manager.download_nim(nim_profile):
        print(f"Could not download NIM {nim_profile}, exiting")
        exit()
    return nim_manager

def start_nim_service(nim_manager, nim_profile):
    http_ports = []
    grpc_ports = []
    if not nim_manager.start_nim(nim_profile, http_ports=http_ports, grpc_ports=grpc_ports, force_nim_start=True):
        print(f"Start NIM fail for {nim_profile}")
        exit()
    assert len(http_ports) == 1 and http_ports[0] == '8000' and len(grpc_ports)==0, "Example only provided for HTTP port 8000."

def create_openai_client():
    return OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

def get_user_input():
    return input("You: ")

def handle_user_exit(user_input):
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Assistant: Goodbye! Have a great day!")
        return True
    return False

def generate_response(client, messages, stream):
    chat_response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=messages,
        max_tokens=2000,
        stream=stream
    )
    return chat_response

def process_response(chat_response, stream):
    assistant_message = ""
    if stream:
        for chunk in chat_response:
            content = chunk.choices[0].delta.content or ""
            print(content, end="")
            assistant_message += content
        print()
    else:
        assistant_message = chat_response.choices[0].message.content
        print("Assistant:", assistant_message)
    return assistant_message

def main():
    print("Welcome to chat using ChatRTX APIs example!")
    print("Type your query and press Enter.")
    print("Type 'exit' or 'quit' to terminate the program.\n")

    nim_profile = "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.0-RTX"

    try:
        nim_manager = initialize_nim_manager(nim_profile)
        start_nim_service(nim_manager, nim_profile)
        client = create_openai_client()
        messages = [
            {"role": "user", "content": "Hello! How are you?"},
            {"role": "assistant", "content": "Hi! I am quite well, how can I help you today?"}
        ]
        _stream = True

        while True:
            user_input = get_user_input()
            if handle_user_exit(user_input):
                break
            messages.append({"role": "user", "content": user_input})
            chat_response = generate_response(client, messages, _stream)
            assistant_message = process_response(chat_response, _stream)
            messages.append({"role": "assistant", "content": assistant_message})

    except Exception as e:
        print(f"An error occurred: {e}\n")
    finally:
        if 'nim_manager' in locals():
            nim_manager.stop_nim(nim_profile)

if __name__ == "__main__":
    main()
