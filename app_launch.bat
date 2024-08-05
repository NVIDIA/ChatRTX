:: SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
:: SPDX-License-Identifier: MIT
::
:: Permission is hereby granted, free of charge, to any person obtaining a
:: copy of this software and associated documentation files (the "Software"),
:: to deal in the Software without restriction, including without limitation
:: the rights to use, copy, modify, merge, publish, distribute, sublicense,
:: and/or sell copies of the Software, and to permit persons to whom the
:: Software is furnished to do so, subject to the following conditions:
::
:: The above copyright notice and this permission notice shall be included in
:: all copies or substantial portions of the Software.
::
:: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
:: IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
:: FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
:: THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
:: LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
:: FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
:: DEALINGS IN THE SOFTWARE.

@echo off
setlocal enabledelayedexpansion

:: Get the absolute path of the parent directory
set "parentDir=%~dp0..\.."
for %%I in ("%parentDir%") do set "absParentDir=%%~fI"

:: Set the environment path
set "env_path=%absParentDir%\env_nvd_rag"

:: Check if the environment exists
if exist "%env_path%\Scripts\python.exe" (
    echo Environment path found: %env_path%
    "%env_path%\Scripts\python.exe" "app_launch.py"
) else (
    echo Environment with 'env_nvd_rag' not found.
    echo Looked in: %env_path%
    pause
)

endlocal
