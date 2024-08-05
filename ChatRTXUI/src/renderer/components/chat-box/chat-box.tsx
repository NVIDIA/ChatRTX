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

import './chat-box.scss'
import React, { useEffect, useRef, useState } from 'react'
import { HistoryItem } from '../../../electron/types'
import { Button, Stack, Typography } from '@mui/material'
import type { ClientAPI } from '../../../electron/preload'
import IconElement from '../icon-element/icon-element'
import { useTranslation } from 'react-i18next'
import { themeSettings } from '../../theme/theme'
import { MuiMarkdown } from 'mui-markdown'
import CustomIconButton from '../custom-icon-button/custom-icon-button'
import CustomModalDialog from '../custom-modal-dialog/custom-modal-dialog'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function ChatBox({ isDrawerOpen }: { isDrawerOpen?: boolean }) {
    const [history, setHistory] = useState<HistoryItem[]>(
        clientAPI.getHistory()
    )
    const container = useRef<HTMLDivElement>(null)
    const { t } = useTranslation()

    const [openClearChatConfirmation, setOpenClearChatConfirmation] =
        useState(false)
    const [waitingForResponse, setWaitingForResponse] = useState(
        clientAPI.isWaiting()
    )

    useEffect(() => {
        scrollToBottom()
        setLinks()
    }, [history])

    useEffect(() => {
        const waitingListener = clientAPI.onWaiting((value: boolean) =>
            setWaitingForResponse(value)
        )
        const onHistoryUpdate = (data: HistoryItem[]) => setHistory(data)
        const historyListener = clientAPI.onHistoryUpdate(onHistoryUpdate)
        return () => {
            historyListener()
            waitingListener()
        }
    }, [])

    let scrollTimer!: any

    let linksTimer!: any

    const scrollToBottom = () => {
        cancelAnimationFrame(scrollTimer)

        scrollTimer = requestAnimationFrame(() => {
            const responses = container.current?.querySelectorAll(
                '.history-item-response-container'
            )

            if (responses && responses.length) {
                responses[responses.length - 1].scrollIntoView({
                    behavior: 'smooth',
                })
            }
        })
    }

    const setLinks = () => {
        cancelAnimationFrame(linksTimer)

        linksTimer = requestAnimationFrame(() => {
            Array.from(container.current?.querySelectorAll('a') || []).forEach(
                (link) => {
                    link.onclick = (event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        clientAPI.openPath(link.dataset['link'])
                    }
                }
            )
        })
    }

    const formatResponse = (text: string) =>
        (text || '').split(' class=').join(' className=')

    return (
        <div
            className="chat-box-wrapper"
            data-isdraweropen={isDrawerOpen}
            ref={container}
        >
            <Stack
                className="chat-box-container"
                direction="column"
                justifyContent="flex-start"
            >
                {history.length === 0 && (
                    <div className="empty-chatbox-message">
                        <div className="nvidia-logo">
                            <svg
                                width="102"
                                height="68"
                                viewBox="0 0 102 68"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    d="M37.9264 20.1897V14.0754C38.5132 14.0333 39.1084 13.9995 39.7204 13.9827C56.3363 13.4598 67.243 28.3448 67.243 28.3448C67.243 28.3448 55.4644 44.7985 42.8474 44.7985C30.2304 44.7985 39.4019 44.5033 37.9348 44.0057V25.469C44.4067 26.2533 45.7062 29.1291 49.596 35.6482L58.2477 28.3111C58.2477 28.3111 51.935 19.9788 41.2881 19.9788C30.6412 19.9788 39.0246 20.0632 37.9348 20.1812M37.9264 0V9.13342C38.5216 9.08282 39.1168 9.04909 39.7204 9.02379C62.8334 8.23948 77.8816 28.0918 77.8816 28.0918C77.8816 28.0918 60.5867 49.2429 42.5708 49.2429C24.5549 49.2429 39.3767 49.0911 37.918 48.8297V54.4716C39.1587 54.6319 40.4498 54.7246 41.7911 54.7246C58.5579 54.7246 70.6802 46.1141 82.4253 35.918C84.3703 37.4867 92.3429 41.2986 93.9776 42.9684C82.811 52.3717 56.7974 59.9449 42.051 59.9449C27.3047 59.9449 39.2593 59.8606 37.918 59.7257V67.6531H101.64V0H37.918H37.9264ZM37.9264 44.0142V48.8297C22.4171 46.0466 18.1165 29.8375 18.1165 29.8375C18.1165 29.8375 25.5609 21.539 37.9264 20.1981V25.4859C37.918 25.4859 37.9096 25.4859 37.9012 25.4859C31.4125 24.7016 26.3406 30.799 26.3406 30.799C26.3406 30.799 29.1825 41.0625 37.9264 44.0226M10.387 29.1291C10.387 29.1291 19.5752 15.4838 37.9264 14.0754V9.12499C17.6051 10.7695 0 28.0834 0 28.0834C0 28.0834 9.96783 57.0776 37.9264 59.7257V54.4632C17.4123 51.8657 10.387 29.1207 10.387 29.1207V29.1291Z"
                                    fill="#76B900"
                                />
                            </svg>
                        </div>
                        <Typography variant="h6">
                            {t('chatboxMessage')}
                        </Typography>
                    </div>
                )}
                {history.map((item, index) => (
                    <Stack
                        key={item.id}
                        className="history-item"
                        direction="column"
                    >
                        <Stack
                            className="history-item-prompt-container"
                            direction={'row'}
                            alignItems={'flex-start'}
                            justifyContent={'flex-start'}
                        >
                            <div className="chat-profile-container">
                                <IconElement
                                    type="account-circle"
                                    size="lg"
                                    fill={true}
                                />
                            </div>
                            <div className="chat-card">
                                <Typography variant="subtitle1">
                                    {t('you')}
                                </Typography>
                                <Typography
                                    className="history-item-prompt-text"
                                    variant="body1"
                                >
                                    {item.prompt}
                                </Typography>
                            </div>
                        </Stack>
                        <Stack
                            className="history-item-response-container"
                            direction={'row'}
                            alignItems={'flex-start'}
                            justifyContent={'flex-start'}
                        >
                            <div className="chat-profile-container">
                                <IconElement
                                    type="chat-rtx-ai-profile"
                                    size="lg"
                                />
                            </div>
                            <Stack
                                direction={'column'}
                                alignItems={'stretch'}
                                justifyContent={'flex-start'}
                                spacing={themeSettings.space.xs}
                            >
                                <div className="chat-card">
                                    <Typography variant="subtitle1">
                                        {'ChatRTX'}
                                    </Typography>
                                    {!!item.response && (
                                        <Stack
                                            direction={'column'}
                                            alignItems={'stretch'}
                                            justifyContent={'flex-start'}
                                            className="chat-response"
                                        >
                                            <MuiMarkdown>
                                                {formatResponse(item.response)}
                                            </MuiMarkdown>
                                        </Stack>
                                    )}
                                    {!item.response && (
                                        <div
                                            data-responding={true}
                                            className="pending-dots"
                                        ></div>
                                    )}
                                </div>
                                {!!item.isResponseGenerated && (
                                    <Stack
                                        className="icons-button-row"
                                        flexDirection={'row'}
                                        alignItems={'center'}
                                        justifyContent={'flex-start'}
                                    >
                                        {item.isTextResponse && (
                                            <CustomIconButton
                                                size={
                                                    themeSettings.iconSizes
                                                        .medium
                                                }
                                                className="chat-icons-button"
                                                type="copy-clipboard"
                                                tooltip={t('copyToClipboard')}
                                                onClick={() => {
                                                    const tempDivElement =
                                                        document.createElement(
                                                            'div'
                                                        )
                                                    tempDivElement.innerHTML =
                                                        item.response
                                                    navigator.clipboard.writeText(
                                                        tempDivElement.textContent ||
                                                            tempDivElement.innerText ||
                                                            ''
                                                    )
                                                }}
                                            ></CustomIconButton>
                                        )}
                                        <CustomIconButton
                                            disabled={waitingForResponse}
                                            size={
                                                themeSettings.iconSizes.medium
                                            }
                                            className="chat-icons-button"
                                            type="regenerate-response"
                                            tooltip={t('regerateResponse')}
                                            onClick={() =>
                                                clientAPI.retryPrompt(item.id)
                                            }
                                        ></CustomIconButton>
                                    </Stack>
                                )}
                            </Stack>
                        </Stack>
                    </Stack>
                ))}
            </Stack>
            {history.length > 0 && (
                <div className="clear-chat-container">
                    <Button
                        className="clear-chat-button"
                        color="transparent"
                        onClick={(event) => setOpenClearChatConfirmation(true)}
                        disabled={waitingForResponse}
                    >
                        <IconElement
                            type="common-clear"
                            size={themeSettings.font.sizes.lg}
                        />
                        <Typography className="clear-chat-text" variant="body2">
                            {t('clearChat')}
                        </Typography>
                    </Button>
                </div>
            )}

            <CustomModalDialog
                dialogHeader={t('clearChatHeader')}
                openDialog={openClearChatConfirmation}
                primaryButtonText={t('CLEAR')}
                secondaryButtonText={t('CANCEL')}
                onPrimaryButtonClick={() => {
                    setOpenClearChatConfirmation(false), clientAPI.resetChat()
                }}
                onSecondaryButtonClick={() =>
                    setOpenClearChatConfirmation(false)
                }
            >
                <Stack
                    flexDirection={'column'}
                    alignItems={'stretch'}
                    justifyContent={'flex-start'}
                >
                    <Typography variant="body2">
                        {t('clearChatDesc')}
                    </Typography>
                </Stack>
            </CustomModalDialog>
        </div>
    )
}
