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



import {
    app,
    BrowserWindow,
    ipcMain,
    screen,
    shell,
    globalShortcut,
} from 'electron'
import { join } from 'path'
import { platform } from 'os'
import ChatBridge from './chat-bridge'
import { BridgeCommands } from '../bridge_commands'
import log from 'electron-log/main'
import path from 'path'
import { TRT_LLM_RAG_DIR } from '../bridge_commands/config'

log.initialize({ spyRendererConsole: true })
Object.assign(console, log.functions)

let mainWindow: null | BrowserWindow
let chatBridge: ChatBridge = undefined
const osPlatform = platform()
const indexPath = join(__dirname, '..', 'index.html')
const systemLanguage = app.getLocale().toLowerCase()

const reload = () => !!mainWindow && mainWindow.reload()
const openDevTools = () => !!mainWindow && mainWindow.webContents.openDevTools()

console.log('systemLanguage', systemLanguage)

const createWindow = () => {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize

    mainWindow = new BrowserWindow({
        width: Math.min(width, 1280),
        height: Math.min(height, 900),
        minHeight: Math.min(height, 900),
        minWidth: Math.min(width, 1280),
        backgroundColor: '#000',
        frame: false,
        webPreferences: {
            preload: join(__dirname, 'preload.js'),
            sandbox: false,
            contextIsolation: true,
            devTools: !app.isPackaged,
        },
    })

    log.transports.file.resolvePathFn = () =>
        app.isPackaged
            ? path.join(
                  process.env.PROGRAMDATA,
                  'NVIDIA Corporation/ChatRTX/logs/main.log'
              )
            : path.join(TRT_LLM_RAG_DIR, 'ChatRTXUI/main.log')

    log.transports.file.getFile()

    let chatBot = undefined
    const chatBotEngine = BridgeCommands(app.isPackaged)
    chatBotEngine
        .then((bridgeCommands) => {
            chatBot = bridgeCommands.Chatbot
            chatBridge = new ChatBridge(mainWindow, chatBot)
            chatBridge.clearStorage()
        })
        .catch((error) => {
            console.log('Error while getting bridge commands', error)
        })

    mainWindow.loadURL(`file://${indexPath}`)
    mainWindow.removeMenu()

    mainWindow.once('ready-to-show', () => {
        if (!app.isPackaged) {
            mainWindow.webContents.openDevTools()
        }
    })

    mainWindow.on('closed', () => {
        mainWindow = null
        if (chatBridge) {
            chatBridge.destroy()
        }
        if (chatBot) {
            chatBot.shutdown().then()
        }
    })

    mainWindow.webContents.session.on('will-download', (event, item) => {
        const extension = item.getMimeType().split('/')
        const filters = []

        if (extension[1] === 'png') {
            filters.push({
                name: 'Image',
                extensions: ['png'],
            })
        }

        item.setSaveDialogOptions({ filters })
    })
}

const gotTheLock = app.requestSingleInstanceLock()

if (!gotTheLock) {
    app.quit()
} else {
    app.on('second-instance', (_event, commandLine) => {
        if (mainWindow) {
            if (mainWindow.isMinimized()) {
                mainWindow.restore()
            }
            mainWindow.focus()
        }
    })
    app.whenReady()
        .then(() => {
            if (!app.isPackaged) {
                const reloadReg = globalShortcut.register(
                    'CommandOrControl+R',
                    reload
                )
                if (!reloadReg) {
                    console.log('Reload shortcut registration failed')
                }

                const devToolsReg = globalShortcut.register(
                    'CommandOrControl+Shift+I',
                    openDevTools
                )
                if (!devToolsReg) {
                    console.log('Dev tools shortcut registration failed')
                }
            }
        })
        .then(() => createWindow())
}

app.on('window-all-closed', () => {
    if (osPlatform !== 'darwin' && osPlatform !== 'sunos') {
        app.quit()
    }
})

const openExternal = (url) => {
    try {
        new URL(url)
        shell.openExternal(url)
    } catch (err) {
        console.log('Invalid URL')
    }
}

ipcMain.handle('minimize', () =>
    mainWindow ? mainWindow.minimize() : undefined
)
ipcMain.handle('close', () => (mainWindow ? mainWindow.close() : undefined))
ipcMain.on('isMaximized', (event) => {
    event.returnValue = mainWindow.isMaximized()
})
ipcMain.handle('maximize', () => mainWindow.maximize())
ipcMain.handle('unmaximize', () => mainWindow.unmaximize())
ipcMain.handle('openLink', (_event, url: string) => openExternal(url))
ipcMain.on('isAppReady', (event) => {
    event.returnValue = !!chatBridge && chatBridge.ready
})
