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

import { HistoryItem } from './types'
import { BrowserWindow, ipcMain } from 'electron'
import { HISTORY_UPDATE_EVENT } from './constants'
import Equals from './equals'

const Events = [HISTORY_UPDATE_EVENT] as const

type EventNames = (typeof Events)[number]

export default class HistoryManager {
    private _win!: BrowserWindow

    private _history: HistoryItem[] = []

    get history() {
        return this._history.map((h) => ({ ...h }))
    }

    set history(items: HistoryItem[]) {
        if (Equals(this._history, items)) {
            return
        }

        this._history = items
        this.emit(HISTORY_UPDATE_EVENT, this._history)
    }

    constructor(win) {
        this._win = win
        this._initIpcEvents()
    }

    clear = () => {
        this.history = []
    }

    undoLast = () => {
        this.history = this.history.slice(0, -1)
    }

    add = (id: string, prompt: string) => {
        this.history = this.history.concat([
            {
                id,
                prompt,
                response: '',
                isResponseGenerated: false,
            },
        ])
    }

    updateResponse = (id: string, response: string) => {
        this.history = this.history.map((item) =>
            item.id !== id ? item : { ...item, response }
        )
    }

    updateResponseGenerated = (id: string, isResponseGenerated: boolean) => {
        this.history = this.history.map((item) =>
            item.id !== id ? item : { ...item, isResponseGenerated }
        )
    }

    updateResponseType = (id: string, isTextResponse: boolean) => {
        this.history = this.history.map((item) =>
            item.id !== id ? item : { ...item, isTextResponse }
        )
    }

    emit = (eventName: EventNames, data?: any) =>
        !!this._win &&
        Events.indexOf(eventName as any) > -1 &&
        this._win.webContents.send(eventName, data)

    private _initIpcEvents() {
        ipcMain.on('getHistory', (event) => {
            event.returnValue = this.history
        })
    }
}
