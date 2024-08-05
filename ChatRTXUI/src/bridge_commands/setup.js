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


const {
    NV_BIN1,
    NV_BIN2,
    NV_BIN3,
    NV_BIN4,
    NV_BIN5,
    NV_RAG,
} = require('./config')
const {
    NV_RAG_PACK,
    NV_BIN1_PACK,
    NV_BIN2_PACK,
    NV_BIN3_PACK,
    NV_BIN4_PACK,
    NV_BIN5_PACK,
} = require('./config-packed')

function pySetup(isPackaged) {
    try {
        if (!isPackaged) {
            process.env.PATH = `${NV_RAG};${NV_BIN1};${NV_BIN2};${NV_BIN3};${NV_BIN4};${NV_BIN5}`
        } else {
            process.env.PATH = `${NV_RAG_PACK};${NV_BIN1_PACK};${NV_BIN2_PACK};${NV_BIN3_PACK};${NV_BIN4_PACK};${NV_BIN5_PACK}`
        }

        process.env['PYTHON_BIN'] = 'python'
        process.env['REQ_TIMEOUT'] = 100000 //ms
        process.env['PYTHONIOENCODING'] = 'utf-8'

        console.log('Conda setup done!')
    } catch (error) {
        console.error('FATAL ERROR: Conda setup error.', error)
    }
}

module.exports = { pySetup: pySetup }
