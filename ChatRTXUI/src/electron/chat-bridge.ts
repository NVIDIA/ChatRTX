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

import { BrowserWindow, ipcMain } from 'electron'
import ID from './id'
import {
    APP_READY,
    ON_PYTHON_ENGINE_INIT,
    ON_PYTHON_ENGINE_INIT_ERROR,
    WAITING_FOR_RESPONSE_UPDATE_EVENT,
} from './constants'
import HistoryManager from './history-manager'
import ModelManager from './model-manager'
import SpeechManager from './speech-manager'
import DatasetManager from './dataset-manager'
import FineTuningManager from './fine-tuning-manager'
import AsrManager from './asr-manager'
import SampleQuestionManager from './sample-question-manager'
import { ModelId } from './types'

const Events = [
    APP_READY,
    ON_PYTHON_ENGINE_INIT,
    ON_PYTHON_ENGINE_INIT_ERROR,
    WAITING_FOR_RESPONSE_UPDATE_EVENT,
] as const

type EventNames = (typeof Events)[number]

export default class ChatBridge {
    private _win!: BrowserWindow

    private _responseTimer!: any

    private _responsePromise!: any

    private _historyManager!: HistoryManager

    private _modelManager!: ModelManager

    private _fineTuningManager!: FineTuningManager

    private _datasetManaget!: DatasetManager

    private _speechManager!: SpeechManager

    private _asrManager!: AsrManager

    private _sampleQuestionManager!: SampleQuestionManager

    private _isWaiting = false

    get isWaiting() {
        return this._isWaiting
    }

    set isWaiting(value: boolean) {
        this._isWaiting = value
        this.emit(WAITING_FOR_RESPONSE_UPDATE_EVENT, this.isWaiting)
    }

    ready = false

    private _chatBot: any

    constructor(win: BrowserWindow, chatBot) {
        this._win = win
        this._chatBot = chatBot
        this._initIpcEvents()
        this.initChatbotEngine()
        this._historyManager = new HistoryManager(win)
        this._modelManager = new ModelManager(win, chatBot)
        this._fineTuningManager = new FineTuningManager(win, chatBot)
        this._datasetManaget = new DatasetManager(win, chatBot)
        this._speechManager = new SpeechManager(win)
        this._asrManager = new AsrManager(chatBot)
        this._sampleQuestionManager = new SampleQuestionManager(chatBot)
    }

    private _initIpcEvents() {
        ipcMain.handle('resetChat', (_event) => this.resetChat())

        ipcMain.handle('undoPrompt', (_event) => this.undoPrompt())

        ipcMain.handle('retryPrompt', (_event, id) => this.retryPrompt(id))

        ipcMain.handle('sendPrompt', (_event, prompt, isStreaming) =>
            this.sendPrompt(prompt, isStreaming)
        )

        ipcMain.on('isWaiting', (event) => {
            event.returnValue = this.isWaiting
        })

        this.handlePythonEvents()
    }

    private handlePythonEvents() {
        this._chatBot.listenPythonEvents(Events).subscribe((value) => {
            const eventName = value?.channel as EventNames
            this.handleEvents(eventName, value?.data)
        })
    }

    private handleEvents(eventName: EventNames, data: any) {
        console.log('In handle events')
        switch (eventName) {
            case ON_PYTHON_ENGINE_INIT:
                this.ready = true
                this.emit(APP_READY)
                break
            case ON_PYTHON_ENGINE_INIT_ERROR:
                this.ready = false
                this.emit(APP_READY)
                break
        }
    }

    private initChatbotEngine() {
        this._chatBot
            .initChatbotEngine()
            .then((res) => {
                console.log('call initChatbotEngine returned response ', res)
            })
            .catch((error) => {
                console.log(
                    'Error occured while calling initChatbotEngine',
                    error
                )
            })
    }

    private _clearResponse = () => {
        clearTimeout(this._responseTimer)

        if (typeof this._responsePromise === 'function') {
            this._responsePromise('')
        }
    }

    clearStorage() {
        this._historyManager.clear()
    }

    resetChat = () => {
        this._clearResponse()
        this.clearStorage()
    }

    undoPrompt = () => {
        this._clearResponse()
        this._historyManager.undoLast()
    }

    retryPrompt = (id: string) => {
        let historyItem = this._historyManager.history.find(
            (value) => value.id === id
        )
        if (!historyItem) {
            return
        }

        this.isWaiting = true

        this._historyManager.updateResponse(historyItem.id, '')
        this._historyManager.updateResponseGenerated(historyItem.id, false)
        this._historyManager.updateResponseType(historyItem.id, false)

        historyItem.prompt

        return this.fetchPrompt(historyItem.id, historyItem.prompt, true)
    }

    sendPrompt = (prompt: string, isStreaming: boolean) => {
        const id = ID()

        this._historyManager.add(id, prompt)

        this._clearResponse()

        this.isWaiting = true

        return this.fetchPrompt(id, prompt, isStreaming)
    }

    private fetchPrompt = (
        id: string,
        prompt: string,
        isStreaming: boolean
    ) => {
        return new Promise((resolve) => {
            this._responsePromise = (resp: string) => {
                resolve(resp)
            }
            this._responsePromise.id = id

            if (!this._responsePromise || this._responsePromise.id !== id) {
                return
            }
            let response = ''
            this._chatBot
                .query(prompt, isStreaming, (res: string) => {
                    response = response + res
                    this._historyManager.updateResponse(id, response)
                    this._responsePromise(response)
                })
                .then(() => {
                    this._historyManager.updateResponseGenerated(id, true)
                    this._historyManager.updateResponseType(
                        id,
                        this._modelManager.model !== ModelId.Clip
                    )
                    if (this.isWaiting) {
                        this.isWaiting = false
                    }
                })
                .catch((error) => {
                    console.error('Error while getting query response ', error)
                    if (this.isWaiting) {
                        this.isWaiting = false
                    }
                })
        })
    }

    emit = (eventName: EventNames, data?: any) =>
        !!this._win && this._win.webContents.send(eventName, data)

    destroy = () => {
        this.ready = false
        /** TODO: shut down services */
    }
}
