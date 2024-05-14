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

import os
import win32process
import win32security

token = win32security.OpenProcessToken(win32process.GetCurrentProcess(),
                                       win32security.TOKEN_ALL_ACCESS)
token_info = win32security.GetTokenInformation(token, win32security.TokenPrivileges)

# Iterate over privileges and output the ones that are enabled
priv_list = []
for priv_id, priv_flags in token_info:
    # Check if privilege is enabled
    if priv_flags == win32security.SE_PRIVILEGE_ENABLED or win32security.SE_PRIVILEGE_ENABLED_BY_DEFAULT:
        # Lookup the name of the privilege
        priv_name = win32security.LookupPrivilegeName(None, priv_id)
        priv_list.append(priv_name)

print(f"Privileges of original process: {priv_list}")

token_r = win32security.CreateRestrictedToken(
            token, win32security.DISABLE_MAX_PRIVILEGE,
            None, None, None)

token_info_r = win32security.GetTokenInformation(token_r, win32security.TokenPrivileges)

# Iterate over privileges and output the ones that are enabled
priv_list = []
for priv_id, priv_flags in token_info_r:
    # Check if privilege is enabled
    if priv_flags == win32security.SE_PRIVILEGE_ENABLED or win32security.SE_PRIVILEGE_ENABLED_BY_DEFAULT:
        # Lookup the name of the privilege
        priv_name = win32security.LookupPrivilegeName(None, priv_id)
        priv_list.append(priv_name)

print(f"Privileges of restricted app token: {priv_list}")

startup = win32process.STARTUPINFO()

(hProcess, hThread,
 processId, threadId) = win32process.CreateProcessAsUser(
    token_r, "app_launch_core.bat", None, None, None,
    True, win32process.NORMAL_PRIORITY_CLASS, None, None, startup)

print("Process Id = ", processId)