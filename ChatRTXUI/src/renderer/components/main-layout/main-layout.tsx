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

import './main-layout.scss'
import React, { useEffect, useState } from 'react'
import NavBar from '../nav-bar/nav-bar'
import StatusHeader from '../status-header/status-header'
import { CircularProgress, Stack, Typography } from '@mui/material'
import Prompt from '../prompt/prompt'
import ChatBox from '../chat-box/chat-box'
import SampleQueryBox from '../sample-query-box/sample-query-box'
import { useTranslation } from 'react-i18next'
import ModelDrawer from '../model-drawer/model-drawer'
import DatasetDrawer from '../dataset-drawer/dataset-drawer'
import type { ClientAPI } from '../../../electron/preload'
import { themeSettings } from '../../theme/theme'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function MainLayout() {
    const [openModelDrawer, setOpenModelDrawer] = useState(false)
    const [openDatasetDrawer, setOpenDatasetDrawer] = useState(false)
    const [isAppReady, setIsAppReady] = useState(clientAPI.isAppReady())

    const { t } = useTranslation()

    const toggleModelDrawer = (newOpen: boolean) => () => {
        setOpenDatasetDrawer(false)
        setOpenModelDrawer(newOpen)
    }

    const toggleDatasetDrawer = (newOpen: boolean) => () => {
        setOpenModelDrawer(false)
        setOpenDatasetDrawer(newOpen)
    }

    useEffect(() => {
        const onAppReadyListener = clientAPI.onAppReady(() =>
            setIsAppReady(true)
        )

        return () => {
            onAppReadyListener()
        }
    }, [])

    return (
        <div className="chat-rtx-app">
            <NavBar />
            {isAppReady && (
                <Stack
                    className="chat-rtx-app-layout"
                    direction="row"
                    alignItems={'center'}
                >
                    <Stack
                        className={
                            openDatasetDrawer || openModelDrawer
                                ? 'drawer-open chat-rtx-primary-container'
                                : 'chat-rtx-primary-container'
                        }
                        direction={'column'}
                    >
                        <div className="chat-header">
                            <StatusHeader
                                isModelDrawerOpen={openModelDrawer}
                                isDatasetDrawerOpen={openDatasetDrawer}
                                onModelHeaderCardClicked={toggleModelDrawer(
                                    !openModelDrawer
                                )}
                                onDatasetHeaderCardClicked={toggleDatasetDrawer(
                                    !openDatasetDrawer
                                )}
                            ></StatusHeader>
                        </div>

                        <Stack
                            className="chat-content"
                            direction={'column'}
                            alignItems={'center'}
                        >
                            <ChatBox
                                isDrawerOpen={
                                    openDatasetDrawer || openModelDrawer
                                }
                            />
                            <Stack
                                direction={'column'}
                                className="chat-interaction-components"
                            >
                                <SampleQueryBox />
                                <Prompt />
                                <Typography
                                    className="footer-text"
                                    variant="body3"
                                >
                                    {t('footerText')}
                                </Typography>
                            </Stack>
                        </Stack>
                    </Stack>
                    <ModelDrawer
                        openDrawer={openModelDrawer}
                        onDrawerClosed={toggleModelDrawer(false)}
                    />
                    <DatasetDrawer
                        openDrawer={openDatasetDrawer}
                        onDrawerClosed={toggleDatasetDrawer(false)}
                    />
                </Stack>
            )}
            {!isAppReady && (
                <Stack
                    className="chat-rxt-loading"
                    direction={'column'}
                    alignItems={'center'}
                    justifyContent={'center'}
                    spacing={themeSettings.space.md}
                >
                    <CircularProgress size={48}></CircularProgress>
                    <Typography className="loading-text" variant="body2">
                        {t('loadingApp')}
                    </Typography>
                </Stack>
            )}
        </div>
    )
}
