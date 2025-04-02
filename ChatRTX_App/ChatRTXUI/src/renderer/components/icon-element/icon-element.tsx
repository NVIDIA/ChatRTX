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

import './icon-element.scss'
import React, { useEffect, useRef } from 'react'
import add from '../../assets/images/common-add.svg'
import close from '../../assets/images/common-close.svg'
import clear from '../../assets/images/common-clear.svg'
import subtract from '../../assets/images/common-subtract.svg'
import copy from '../../assets/images/common-copy-generic.svg'
import square from '../../assets/images/shapes-shape-square.svg'
import minimizedMode from '../../assets/images/minimized-mode.svg'
import chevronRight from '../../assets/images/shapes-chevron-right.svg'
import chevronDown from '../../assets/images/shapes-chevron-down.svg'
import mic from '../../assets/images/microphone.svg'
import trash from '../../assets/images/common-trash.svg'
import refresh from '../../assets/images/common-refresh.svg'
import undo from '../../assets/images/common-undo.svg'
import info from '../../assets/images/common-info-circle.svg'
import error from '../../assets/images/common-error.svg'
import check from '../../assets/images/common-check-circle.svg'
import warning from '../../assets/images/common-warning.svg'
import download from '../../assets/images/download.svg'
import sync from '../../assets/images/common-sync.svg'
import cog from '../../assets/images/common-cog.svg'
import pencil from '../../assets/images/editor-pencil.svg'
import folder from '../../assets/images/files-folder-closed.svg'
import chatRtxAIProfile from '../../assets/images/chat-rtx-ai-profile.svg'
import accountCircle from '../../assets/images/account-circle.svg'
import checkBoxOn from '../../assets/images/checkbox-on.svg'
import checkBoxOff from '../../assets/images/checkbox-off.svg'
import menu3Dots from '../../assets/images/menu-3-dots.svg'
import copyClipboard from '../../assets/images/copy-clipboard.svg'
import regenerateResponse from '../../assets/images/regenerate-response.svg'
import drawerClose from '../../assets/images/drawer-close.svg'
import send from '../../assets/images/send.svg'
import { themeSettings } from '../../theme/theme'
import { Stack } from '@mui/material'

const Icons = {
    'account-circle': decodeURIComponent(accountCircle).split(
        'data:image/svg+xml,'
    )[1],
    microphone: decodeURIComponent(mic).split('data:image/svg+xml,')[1],
    'chevron-right': decodeURIComponent(chevronRight).split(
        'data:image/svg+xml,'
    )[1],
    'common-check-circle': decodeURIComponent(check).split(
        'data:image/svg+xml,'
    )[1],
    'common-add': decodeURIComponent(add).split('data:image/svg+xml,')[1],
    'common-close': decodeURIComponent(close).split('data:image/svg+xml,')[1],
    'common-clear': decodeURIComponent(clear).split('data:image/svg+xml,')[1],
    'common-cog': decodeURIComponent(cog).split('data:image/svg+xml,')[1],
    'common-copy-generic': decodeURIComponent(copy).split(
        'data:image/svg+xml,'
    )[1],
    'common-error': decodeURIComponent(error).split('data:image/svg+xml,')[1],
    'common-info-circle': decodeURIComponent(info).split(
        'data:image/svg+xml,'
    )[1],
    'common-refresh': decodeURIComponent(refresh).split(
        'data:image/svg+xml,'
    )[1],
    'common-subtract': decodeURIComponent(subtract).split(
        'data:image/svg+xml,'
    )[1],
    'common-sync': decodeURIComponent(sync).split('data:image/svg+xml,')[1],
    'common-trash': decodeURIComponent(trash).split('data:image/svg+xml,')[1],
    'common-undo': decodeURIComponent(undo).split('data:image/svg+xml,')[1],
    'common-warning': decodeURIComponent(warning).split(
        'data:image/svg+xml,'
    )[1],
    'common-download': decodeURIComponent(download).split(
        'data:image/svg+xml,'
    )[1],
    'copy-clipboard': decodeURIComponent(copyClipboard).split(
        'data:image/svg+xml,'
    )[1],
    'regenerate-response': decodeURIComponent(regenerateResponse).split(
        'data:image/svg+xml,'
    )[1],
    'editor-pencil': decodeURIComponent(pencil).split('data:image/svg+xml,')[1],
    'shapes-chevron-down': decodeURIComponent(chevronDown).split(
        'data:image/svg+xml,'
    )[1],
    'shapes-shape-square': decodeURIComponent(square).split(
        'data:image/svg+xml,'
    )[1],
    'minimized-mode': decodeURIComponent(minimizedMode).split(
        'data:image/svg+xml,'
    )[1],
    'chat-rtx-ai-profile': decodeURIComponent(chatRtxAIProfile).split(
        'data:image/svg+xml,'
    )[1],
    'checkbox-on': decodeURIComponent(checkBoxOn).split(
        'data:image/svg+xml,'
    )[1],
    'checkbox-off': decodeURIComponent(checkBoxOff).split(
        'data:image/svg+xml,'
    )[1],
    'menu-3dots': decodeURIComponent(menu3Dots).split('data:image/svg+xml,')[1],
    'drawer-close': decodeURIComponent(drawerClose).split(
        'data:image/svg+xml,'
    )[1],
    send: decodeURIComponent(send).split('data:image/svg+xml,')[1],
    folder: decodeURIComponent(folder).split('data:image/svg+xml,')[1],
}

export type IconsTypes = keyof typeof Icons

const icons = (type: IconsTypes) => Icons[type] || ''

const parseSize = (size: string) => {
    if (!size) {
        return '1rem'
    }
    return (themeSettings.font.sizes as any)[`${size}`] || size
}

const parseColor = (color: keyof typeof themeSettings.colors) =>
    themeSettings.colors[`${color}`]

export default function IconElement({
    size,
    type,
    color,
    fill,
    opacity,
}: {
    size?: string
    fill?: boolean
    color?: keyof typeof themeSettings.colors
    type: IconsTypes
    opacity?: number
}) {
    const container = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (container.current) {
            container.current.innerHTML = icons(type)
        }
    }, [])

    return (
        <Stack
            className="icon-element"
            ref={container}
            style={{
                fontSize: parseSize(size),
                color: parseColor(color),
            }}
            data-fill={fill}
            alignItems="center"
            justifyContent="center"
        ></Stack>
    )
}
