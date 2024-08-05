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

import './nav-bar.scss'
import React, { useState } from 'react'
import { Stack, Typography } from '@mui/material'
import type { ClientAPI } from '../../../electron/preload'
import IconElement from '../icon-element/icon-element'
import { themeSettings } from '../../theme/theme'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function NavBar() {
    const [maximized, setMaximized] = useState<boolean>(clientAPI.isMaximized())

    const minimize = () => {
        if (!clientAPI) {
            return
        }
        clientAPI.minimize()
    }

    const maximize = () => {
        if (!clientAPI) {
            return
        }

        if (clientAPI.isMaximized()) {
            clientAPI.unmaximize()
        } else {
            clientAPI.maximize()
        }

        setMaximized(clientAPI.isMaximized())
    }

    const close = () => {
        if (!clientAPI) {
            return
        }
        clientAPI.close()
    }

    return (
        <div id="nav-bar">
            <Stack
                className="drag-region"
                alignItems="center"
                justifyContent="space-between"
                direction="row"
            >
                <Stack
                    className="window-title"
                    alignItems="center"
                    direction="row"
                    gap={themeSettings.space.sm}
                >
                    <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path d="M16 0H0V16H16V0Z" fill="#1E1E1E" />
                        <path
                            d="M6.09431 6.47505V5.78442C6.15994 5.77817 6.22869 5.77505 6.29744 5.77505C8.18806 5.71567 9.42869 7.40005 9.42869 7.40005C9.42869 7.40005 8.08806 9.26255 6.65056 9.26255C5.21306 9.26255 6.25681 9.22817 6.09119 9.17192V7.07505C6.82869 7.16255 6.97556 7.49067 7.41931 8.22817L8.40369 7.39692C8.40369 7.39692 7.68494 6.45317 6.47244 6.45317C5.25994 6.45317 6.21306 6.46255 6.09119 6.47505H6.09431ZM6.09119 4.19067V5.22505C6.15994 5.2188 6.22556 5.21567 6.29431 5.21255C8.92556 5.12505 10.6381 7.3688 10.6381 7.3688C10.6381 7.3688 8.66931 9.76255 6.61931 9.76255C4.56931 9.76255 6.25681 9.7438 6.09119 9.71567V10.3532C6.23181 10.3719 6.37869 10.3813 6.53181 10.3813C8.44119 10.3813 9.81932 9.4063 11.1568 8.25317C11.3787 8.4313 12.2849 8.86255 12.4724 9.05005C11.2006 10.1125 8.24119 10.9719 6.56306 10.9719C4.88494 10.9719 6.24431 10.9625 6.09431 10.9469V11.8438H13.3474V4.19067H6.09431H6.09119ZM6.09119 9.17192V9.71567C4.32556 9.40005 3.83494 7.56567 3.83494 7.56567C3.83494 7.56567 4.68181 6.62817 6.09119 6.47505V7.07192C5.35369 6.98442 4.77556 7.67192 4.77556 7.67192C4.77556 7.67192 5.09744 8.83442 6.09431 9.1688L6.09119 9.17192ZM2.95994 7.48442C2.95994 7.48442 4.00681 5.94067 6.09431 5.7813V5.22192C3.78181 5.4063 1.77869 7.36567 1.77869 7.36567C1.77869 7.36567 2.91306 10.6469 6.09431 10.9469V10.3532C3.75994 10.0594 2.95994 7.48442 2.95994 7.48442Z"
                            fill="#76B900"
                        />
                    </svg>

                    <Typography variant="body2" className="body-text-color">
                        NVIDIA ChatRTX
                    </Typography>
                </Stack>

                <Stack
                    className="window-controls"
                    alignItems="center"
                    justifyContent="flex-end"
                    direction="row"
                    gap={themeSettings.space.sm}
                >
                    <Stack
                        className="button min-button"
                        onClick={minimize}
                        alignItems="center"
                        justifyContent="center"
                    >
                        <IconElement type="common-subtract" size="md" />
                    </Stack>

                    <Stack
                        className="button max-button"
                        onClick={maximize}
                        alignItems="center"
                        justifyContent="center"
                    >
                        {maximized && (
                            <IconElement
                                type="common-copy-generic"
                                color="n050"
                                size="md"
                            />
                        )}
                        {!maximized && (
                            <IconElement
                                type="minimized-mode"
                                size={themeSettings.font.sizes.ms}
                            />
                        )}
                    </Stack>

                    <Stack
                        className="button close-button"
                        onClick={close}
                        alignItems="center"
                        justifyContent="center"
                    >
                        <IconElement type="common-close" size="md" />
                    </Stack>
                </Stack>
            </Stack>
        </div>
    )
}
