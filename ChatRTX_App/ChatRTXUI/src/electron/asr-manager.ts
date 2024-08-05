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

import { ipcMain } from 'electron'
import { writeFileSync, existsSync, mkdirSync } from 'fs'
import path from 'path'
import { app } from 'electron'

export default class AsrManager {
    private _chatBot: any

    constructor(chatBot) {
        this._chatBot = chatBot
        this._initIpcEvents()
    }

    private _initIpcEvents() {
        ipcMain.handle('initAsrModel', (_event) => this.handleInitAsrModel())

        ipcMain.handle('getTextFromAudio', (_event, data) =>
            this.handleRecordedAudio(data)
        )
    }

    handleInitAsrModel = () => {
        this._chatBot
            .initAsrModel()
            .then((value) => {
                console.log('initAsrModel call returned ', value)
            })
            .catch((error) => {
                console.log('initAsrModel returned error ', error)
            })
    }

    handleRecordedAudio = (data: string) => {
        return new Promise((resolve) => {
            const tempDir = path.join(app.getPath('temp'), 'chat-with-rtx-asr')

            if (!existsSync(tempDir)) {
                mkdirSync(tempDir)
            }

            const audioFilePath = path.join(tempDir, 'recordedAudio.wav')

            writeFileSync(
                audioFilePath,
                Buffer.from(data.split(',')[1], 'base64')
            )

            this._chatBot
                .getTextFromAudio(audioFilePath)
                .then((result) => {
                    console.log(`Got text for audio ${result}`)
                    resolve(result)
                })
                .catch((error) => {
                    console.error('Error while getting text from audio ', error)
                    resolve('')
                })
        })
    }
}
