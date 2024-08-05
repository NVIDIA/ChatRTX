'use strict'
/* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


const { join } = require('path')

const trtllmragwindows = join(__dirname, '..', '..', '..')

// Updated below python enviroment path for non packaged app (app built using npm run start-electron)
const NV_RAG = join(trtllmragwindows, '..', '..', 'env_nvd_rag')

module.exports = {
    TRT_LLM_RAG_DIR: trtllmragwindows,
    NV_RAG: NV_RAG,
    NV_RAG_ENGINE: join(
        trtllmragwindows,
        'ChatRTXUI',
        'engine',
        'ChatRTXUIEngine.py'
    ),
    NV_BIN1: join(NV_RAG, 'Library', 'mingw-w64', 'bin'),
    NV_BIN2: join(NV_RAG, 'Library', 'usr', 'bin'),
    NV_BIN3: join(NV_RAG, 'Library', 'bin'),
    NV_BIN4: join(NV_RAG, 'bin'),
    NV_BIN5: join(NV_RAG, 'Scripts'),
}
