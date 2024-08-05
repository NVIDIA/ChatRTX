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
import {
    ModelId,
    SampleQuestionInfo,
    sampleQuestionInfoKey,
    sampleQuestionVal,
} from './types'
import { ON_PYTHON_ENGINE_INIT, ON_PYTHON_ENGINE_INIT_ERROR } from './constants'

const Events = [ON_PYTHON_ENGINE_INIT, ON_PYTHON_ENGINE_INIT_ERROR] as const

type EventNames = (typeof Events)[number]

export default class SampleQuestionManager {
    private _chatBot: any
    private sampleQuestionInfo: SampleQuestionInfo

    constructor(chatBot) {
        this._chatBot = chatBot
        this._initIpcEvents()
    }

    private _initIpcEvents() {
        ipcMain.on('getSampleQuestions', (event, modelId, selected_path) => {
            event.returnValue = this.evalSampleQuestions(modelId, selected_path)
        })
        this.handlePythonEvents()
    }

    private handlePythonEvents() {
        this._chatBot.listenPythonEvents(Events).subscribe((value) => {
            const eventName = value?.channel as EventNames
            this.handleEvents(eventName, value?.data)
        })
    }

    private handleEvents(eventName: EventNames, data) {
        switch (eventName) {
            case ON_PYTHON_ENGINE_INIT:
            case ON_PYTHON_ENGINE_INIT_ERROR:
                this.fetchSampleQuestionInfo()
                break
        }
    }

    private evalSampleQuestions(modeld: ModelId, selected_path: string) {
        let questions: string[] = []
        Object.keys(sampleQuestionInfoKey).forEach((key) => {
            const sampleQuestionVal: sampleQuestionVal =
                this.sampleQuestionInfo[sampleQuestionInfoKey[key]]
            if (
                sampleQuestionVal &&
                sampleQuestionVal.modelIdList.includes(modeld) &&
                sampleQuestionVal.dataset_path === selected_path
            ) {
                questions = sampleQuestionVal.queries
            }
        })
        return questions
    }

    private fetchSampleQuestionInfo() {
        this._chatBot
            .getSampleQuestionInfo()
            .then((value) => {
                console.log('getSampelQuestionInfo returned ', value)
                this.sampleQuestionInfo = JSON.parse(value)
            })
            .catch((error) => {
                console.log('Error while getSampleQuestionInfo ', error)
            })
    }
}
