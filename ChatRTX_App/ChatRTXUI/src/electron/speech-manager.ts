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
import {
    ON_INIT_SPEECH,
    ON_PROCESSING_SPEECH,
    ON_RECORDING,
    ON_RECORDING_TIME,
    ON_SPEECH_ERROR,
    ON_SPEECH_RESULT,
} from './constants'
import Equals from './equals'

const Events = [
    ON_INIT_SPEECH,
    ON_PROCESSING_SPEECH,
    ON_RECORDING,
    ON_RECORDING_TIME,
    ON_SPEECH_ERROR,
    ON_SPEECH_RESULT,
] as const

type EventNames = (typeof Events)[number]

const Instance = {
    results: [] as string[],
    running: false,
    stopCB: () => {},
    messageCB: (str: string) => {},
    readyCB: () => {},
    errorCB: (str: string) => {},
    start: () => {
        Instance.results = []
        Instance.running = true

        setTimeout(() => {
            if (Instance.running === false) {
                return
            }

            Instance.readyCB()

            setTimeout(() => {
                if (Instance.running === false) {
                    return
                }

                // if (Math.random() >= 0.5) {
                //     Instance.errorCB('Oops, random error')
                //     Instance.stop()
                // } else {
                Instance.results.push(new Date().toString())

                setTimeout(() => {
                    if (Instance.running === false) {
                        return
                    }

                    Instance.results.push(`\n${new Date().toString()}`)
                    Instance.stop()
                }, 5000)
                // }
            }, 5000)
        }, 2000)
    },

    stop: () => {
        Instance.running = false

        if (Instance.results.length) {
            Instance.messageCB(Instance.results.join('') || '')
        }

        Instance.stopCB()
        Instance.results = []
    },
}

export default class SpeechManager {
    private _win!: BrowserWindow

    private _isRecording = false

    get isRecording() {
        return this._isRecording
    }

    set isRecording(value: boolean) {
        if (this._isRecording === value) {
            return
        }

        this._isRecording = value
        this.emit(ON_RECORDING, this.isRecording)
    }

    private _isProcessingSpeech = false

    get isProcessingSpeech() {
        return this._isProcessingSpeech
    }

    set isProcessingSpeech(value: boolean) {
        if (this._isProcessingSpeech === value) {
            return
        }

        this._isProcessingSpeech = value
        this.emit(ON_PROCESSING_SPEECH, this.isProcessingSpeech)
    }

    private _isInitializingSpeech = false

    get isInitializingSpeech() {
        return this._isInitializingSpeech
    }

    set isInitializingSpeech(value: boolean) {
        if (this._isInitializingSpeech === value) {
            return
        }

        this._isInitializingSpeech = value
        this.emit(ON_INIT_SPEECH, this.isInitializingSpeech)
    }

    private _recordingTime = 0

    get recordingTime() {
        return this._recordingTime
    }

    set recordingTime(value: number) {
        if (this._recordingTime === value) {
            return
        }

        this._recordingTime = value
        this.emit(ON_RECORDING_TIME, this.recordingTime)
    }

    private _error: string = ''

    get error() {
        return this._error
    }

    set error(value: string) {
        if (this._error === value) {
            return
        }

        this._error = value
        this.emit(ON_SPEECH_ERROR, this._error)
    }

    private _result: string[] = []

    get result() {
        return this._result
    }

    set result(value: string[]) {
        if (Equals(this._result, value)) {
            return
        }

        this._result = value
        this.emit(ON_SPEECH_RESULT, this.result)
    }

    constructor(win: BrowserWindow) {
        this._win = win
        this._initEvents()
    }

    private _initEvents() {
        ipcMain.on('isRecordingSpeech', (event) => {
            event.returnValue = this.isRecording
        })

        ipcMain.on('isProcessingSpeech', (event) => {
            event.returnValue = this.isProcessingSpeech
        })

        ipcMain.on('isInitializingSpeech', (event) => {
            event.returnValue = this.isInitializingSpeech
        })

        ipcMain.on('recordingTime', (event) => {
            event.returnValue = this.recordingTime
        })

        ipcMain.on('speechResult', (event) => {
            event.returnValue = this.result
        })

        ipcMain.on('speechError', (event) => {
            event.returnValue = this.error
        })

        ipcMain.handle('startSpeech', (_event) => this.start())

        ipcMain.handle('endSpeech', (_event) => this.end())
    }

    start = () => {
        this.isInitializingSpeech = true
        this.isRecording = true
        this.result = []
        this.recordingTime = 0
        this.error = ''

        let timer!: any

        const startTimer = () => {
            timer = setInterval(() => {
                this.recordingTime = this.recordingTime + 1
            }, 1000)
        }

        Instance.messageCB = (str: string) => {
            this.result = this.result.concat([str])
        }
        Instance.errorCB = (str: string) => {
            this.error = str
            this.end()
        }
        Instance.readyCB = () => {
            this.isInitializingSpeech = false
            startTimer()
        }
        Instance.stopCB = () => {
            this.isInitializingSpeech = false
            this.isRecording = false
            clearInterval(timer)

            if (this.result.length) {
                this.isProcessingSpeech = true
                setTimeout(() => {
                    this.isProcessingSpeech = false
                }, 2000)
            }
        }

        Instance.start()
    }

    end = () => {
        Instance.stop()
    }

    emit = (eventName: EventNames, data?: any) =>
        !!this._win &&
        Events.indexOf(eventName as any) > -1 &&
        this._win.webContents.send(eventName, data)
}
