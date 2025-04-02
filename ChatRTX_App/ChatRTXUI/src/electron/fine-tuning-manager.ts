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

import { FineTuningInfo, FineTuningProfileConfig, ModelId } from './types'
import { BrowserWindow, dialog, ipcMain } from 'electron'
import {
    FINE_TUNE_UPDATED,
    ON_FINE_TUNING_FOLDER_PATH_UPDATE,
    ON_BASE_MODEL_DOWNLOADED,
    ON_BASE_MODEL_DOWNLOAD_ERROR,
    ON_PROFILE_CREATED,
    ON_PROFILE_CREATE_ERROR,
    ON_PROFILE_SELECTED,
    ON_PROFILE_SELECT_ERROR,
    ON_PROFILE_DELETED,
    ON_PROFILE_DELETED_ERROR,
    ON_FINE_TUNING_ENABLE_ERROR,
    ON_FINE_TUNING_DISABLE_ERROR,
    ON_FINE_TUNING_ENABLED,
    ON_FINE_TUNING_DISABLED,
    ON_PYTHON_ENGINE_INIT,
    ON_PYTHON_ENGINE_INIT_ERROR,
} from './constants'
import Equals from './equals'

const Events = [
    FINE_TUNE_UPDATED,
    ON_FINE_TUNING_FOLDER_PATH_UPDATE,
    ON_BASE_MODEL_DOWNLOADED,
    ON_BASE_MODEL_DOWNLOAD_ERROR,
    ON_PROFILE_CREATED,
    ON_PROFILE_CREATE_ERROR,
    ON_PROFILE_SELECTED,
    ON_PROFILE_SELECT_ERROR,
    ON_PROFILE_DELETED,
    ON_PROFILE_DELETED_ERROR,
    ON_FINE_TUNING_ENABLED,
    ON_FINE_TUNING_DISABLED,
    ON_FINE_TUNING_ENABLE_ERROR,
    ON_FINE_TUNING_DISABLE_ERROR,
    ON_PYTHON_ENGINE_INIT,
    ON_PYTHON_ENGINE_INIT_ERROR,
] as const

type EventNames = (typeof Events)[number]

export default class FineTuningManager {
    private _win!: BrowserWindow
    private _chatbot!: any
    private _fineTuningInfo: FineTuningInfo

    get fineTuningInfo(): FineTuningInfo {
        return this._fineTuningInfo
    }

    set fineTuningInfo(value: FineTuningInfo) {
        if (Equals(value, this._fineTuningInfo)) {
            return
        }
        this._fineTuningInfo = value
        this.emit(FINE_TUNE_UPDATED)
    }

    private _folderPath = ''

    get folderPath() {
        return this._folderPath
    }

    set folderPath(value: string) {
        if (this._folderPath === value) {
            return
        }
        this._folderPath = value
        this.emit(ON_FINE_TUNING_FOLDER_PATH_UPDATE, value)
    }

    constructor(win, chatbot) {
        this._win = win
        this._chatbot = chatbot
        this._initIpcEvents()
    }

    private _initIpcEvents() {
        ipcMain.on('isFineTuningEnabled', (event, parentId) => {
            event.returnValue =
                this.fineTuningInfo[parentId]?.isFineTuningEnabled ?? false
        })

        ipcMain.on('getModelFineTuningDetails', (event, parentId) => {
            event.returnValue = this.fineTuningInfo
                ? this.fineTuningInfo[parentId]
                : {}
        })

        ipcMain.on('getDefaultAdvancedParams', (event, parentId) => {
            event.returnValue = this.fineTuningInfo
                ? this.fineTuningInfo[parentId]?.defaultAdvancedParams
                : {}
        })

        ipcMain.on(
            'getFineTunningProfileConfig',
            (event, parentId, profileId) => {
                event.returnValue = this.fineTuningInfo
                    ? this.fineTuningInfo[
                          parentId
                      ].FineTuningProfileConfigs.find(
                          (profile) => profile.profileId === profileId
                      )
                    : {}
            }
        )

        ipcMain.handle('downloadBaseModel', (_event, parentId) =>
            this.downloadBaseModel(parentId)
        )
        ipcMain.handle('selectFineTuningModel', (_event, parentId, profileId) =>
            this.selectFineTuningModel(parentId, profileId)
        )
        ipcMain.handle(
            'createFineTuningProfile',
            (_event, parentId, fineTuneConfig) =>
                this.createFineTuningProfile(parentId, fineTuneConfig)
        )

        ipcMain.handle(
            'deleteFineTuningProfile',
            (_event, parentId, profileId) =>
                this.deleteFineTuningProfile(parentId, profileId)
        )

        ipcMain.handle('selectFolder', async () => await this.selectFolder())

        ipcMain.handle(
            'toggleFineTuningEnablement',
            (_event, parentId, enabled) =>
                this.toggleFineTuningEnablement(parentId, enabled)
        )

        this.handlePythonEvents()
    }

    private handlePythonEvents() {
        this._chatbot.listenPythonEvents(Events).subscribe((value) => {
            const eventName = value?.channel as EventNames
            this.handleEvents(eventName, value?.data)
        })
    }

    private handleEvents(eventName: EventNames, data) {
        switch (eventName) {
            case ON_PROFILE_CREATED:
                this.emit(ON_PROFILE_CREATED, data as string)
            case ON_PYTHON_ENGINE_INIT:
            case ON_PYTHON_ENGINE_INIT_ERROR:
            case ON_PROFILE_SELECTED:
            case ON_PROFILE_CREATE_ERROR:
            case ON_PROFILE_SELECT_ERROR:
            case ON_PROFILE_DELETED_ERROR:
            case ON_PROFILE_DELETED:
            case ON_BASE_MODEL_DOWNLOADED:
            case ON_FINE_TUNING_ENABLED:
            case ON_FINE_TUNING_DISABLED:
            case ON_FINE_TUNING_DISABLE_ERROR:
            case ON_FINE_TUNING_ENABLE_ERROR:
                this.getFineTuningInfo()
                break
        }
        this.emit(eventName)
    }

    private getFineTuningInfo() {
        this._chatbot
            .getFineTuningInfo()
            .then((value) => {
                if (!value) {
                    console.log('getFineTuningInfo call returned empty', value)
                } else {
                    console.info('getFineTuningInfo call returned with value')
                }
                const fineTuningInfo = JSON.parse(value)
                this.fineTuningInfo = fineTuningInfo
            })
            .catch((error) => {
                console.log('Error for getFineTuningInfo call ', error)
            })
    }

    private downloadBaseModel = (parentId: ModelId) => {
        this._chatbot
            .downloadBaseModel(parentId)
            .then((res) => {
                console.log('Response for call downloadBaseModel ', res)
            })
            .catch((error) => {
                console.log('Error while calling downloadBaseModel ', error)
                this.emit(ON_BASE_MODEL_DOWNLOAD_ERROR)
            })
    }

    private selectFineTuningModel = (parentId: ModelId, profileId: string) => {
        this._chatbot
            .selectFineTuningModel(parentId, profileId)
            .then((res) => {
                console.log('Response for call selectFineTuningModel ', res)
            })
            .catch((error) => {
                console.log('Error while calling selectFineTuningModel ', error)
                this.emit(ON_PROFILE_SELECT_ERROR)
            })
    }

    private createFineTuningProfile = (
        parentId: ModelId,
        profileConfig: FineTuningProfileConfig
    ) => {
        this._chatbot
            .createFineTuningModel(parentId, profileConfig)
            .then((res) => {
                console.log('Response for call createFineTuningModel ', res)
            })
            .catch((error) => {
                console.log('Error while calling createFineTuningModel ', error)
                this.emit(ON_PROFILE_CREATE_ERROR)
            })
    }

    private deleteFineTuningProfile = (
        parentId: ModelId,
        profileId: string
    ) => {
        this._chatbot
            .deleteFineTuningProfile(parentId, profileId)
            .then((res) => {
                console.log('Response for call deletedFineTuningProfile ', res)
            })
            .catch((error) => {
                console.log(
                    'Error while calling deletedFineTuningProfile ',
                    error
                )
                this.emit(ON_PROFILE_DELETED_ERROR)
            })
    }

    private toggleFineTuningEnablement = (
        parentId: ModelId,
        shouldEnable: boolean
    ) => {
        this._chatbot
            .toggeFineTuningEnablement(parentId, shouldEnable)
            .then((res) => {
                console.log('Response for call toggeFineTuningEnablement ', res)
            })
            .catch((error) => {
                console.log(
                    'Error while calling toggeFineTuningEnablement ',
                    error
                )
                this.emit(
                    shouldEnable
                        ? ON_FINE_TUNING_ENABLE_ERROR
                        : ON_FINE_TUNING_DISABLE_ERROR
                )
            })
    }

    private selectFolder = async () => {
        const { canceled, filePaths } = await dialog.showOpenDialog(this._win, {
            properties: ['openDirectory'],
        })
        if (canceled) {
            return
        } else {
            return filePaths[0]
        }
    }

    private emit = (eventName: EventNames, data?: any) =>
        !!this._win &&
        Events.indexOf(eventName as any) > -1 &&
        this._win.webContents.send(eventName, data)
}
