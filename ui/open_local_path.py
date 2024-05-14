# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tkinter as tk
from tkinter import filedialog
import sys
import platform
import os

allowed_extensions = [".txt", ".pdf", ".doc", ".png", ".jpeg", ".jpg", ".bmp", ".docx"]

def open_local_link(root):
    top = tk.Toplevel(root)
    top.attributes('-topmost', True)
    top.withdraw()
    parent = top if platform.system() == "Windows" else None
    initial_path = ''
    arguments = sys.argv[1:]
    if len(arguments) > 0:
        initial_path = arguments[0]

    if (os.path.isdir(initial_path)):
        try:
            os.startfile(initial_path)
            print("Starfile executed to open folder")
        except Exception as e:
            print("Startfile failed for folder, relying on askopenfiles ", e)
            filedialog.askopenfiles(parent=parent, initialdir=initial_path)
            print("askopenfiles executed to open folder")
    elif os.path.isfile(initial_path):
        try:
            file_extension = os.path.splitext(initial_path)[1]
            if file_extension.lower() in allowed_extensions:
                os.startfile(initial_path)
                print("Starfile executed to open file")
            else:
                raise Exception("File type not supported")
        except Exception as e:
            print("Startfile failed to open file ", e)
    else:
        print("Path not idenditfied as folder/file ", initial_path)
    top.destroy()
    return None

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    open_local_link(root)
    root.destroy()
    sys.exit(0)