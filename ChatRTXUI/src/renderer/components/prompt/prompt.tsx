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

import './prompt.scss'
import React, { useEffect, useState } from 'react'
import {
    Button,
    CircularProgress,
    Snackbar,
    Stack,
    Tooltip,
    Typography,
} from '@mui/material'
import { themeSettings } from '../../theme/theme'
import IconElement from '../icon-element/icon-element'
import type { ClientAPI } from '../../../electron/preload'
import { useTranslation } from 'react-i18next'
import AudioRecorder from '../../utils/audio-recorder'
import TextArea from '../text-area/text-area'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function Prompt() {
    const [promptString, setPromptString] = useState('')
    const [isRecordingSpeech, setRecordingSpeech] = useState(
        AudioRecorder.isRecording
    )
    const [isProcessingSpeech, setProcessingSpeech] = useState(
        AudioRecorder.isProcessing
    )
    const [recordingTime, setRecordingTime] = useState(
        AudioRecorder.recordingTime
    )
    const [showMicInitialization, setMicInitialization] = useState(false)

    const [waitingForResponse, setWaitingForResponse] = useState(
        // false
        clientAPI.isWaiting()
    )
    const [toastMessage, setToastMessage] = useState<string>(null)
    const { t } = useTranslation()

    useEffect(() => {
        const waitingListener = clientAPI.onWaiting((value: boolean) =>
            setWaitingForResponse(value)
        )

        const listenForRecoredEvents = AudioRecorder.listenForEvents(
            (evt: string, value: any) => {
                switch (evt) {
                    case 'recording':
                        if (value) {
                            clientAPI.initAsrModel()
                        }
                        setRecordingSpeech(value)
                        break
                    case 'processing':
                        setProcessingSpeech(value)
                        break
                    case 'time':
                        setRecordingTime(value)
                        break
                    case 'asrResponse':
                        setPromptString(value)
                        break
                    case 'error':
                        setToastMessage(t('audioCaptureError'))
                        break
                }
            }
        )

        return () => {
            waitingListener()
            listenForRecoredEvents()
        }
    }, [])

    const onPrompt = () => {
        if (promptString.trim() === '') {
            return
        }
        const isSteaming = true
        clientAPI.sendPrompt(promptString.trim(), isSteaming)
        setPromptString('')
    }

    const onKeyup = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (
            e.key === 'Enter' &&
            !e.ctrlKey &&
            !e.shiftKey &&
            !e.altKey &&
            promptString.trim() !== ''
        ) {
            e.preventDefault()
            e.stopPropagation()
            onPrompt()
        }
    }

    const onMicButtonClick = () => {
        if (AudioRecorder.canRecord && AudioRecorder.isInitialized) {
            return AudioRecorder.toggleRecording()
        }

        if (AudioRecorder.canRecord && !AudioRecorder.isInitialized) {
            setMicInitialization(true)
            AudioRecorder.startStream()
            setTimeout(() => {
                AudioRecorder.isInitialized = true
                setMicInitialization(false)
                onMicButtonClick()
            }, 1000)
            return
        }

        setToastMessage(t('micNotDetected'))
    }

    const handleToastClose = (_: any, reason: string) => {
        if (reason === 'clickaway') {
            return
        }

        setToastMessage(null)
    }

    const disabled =
        isProcessingSpeech || waitingForResponse || showMicInitialization
    const promptDisabled = disabled || isRecordingSpeech

    return (
        <Stack
            alignItems="flex-end"
            direction="row"
            id="chat-form-inputs"
            data-is-recording={isRecordingSpeech}
        >
            <div id="prompt-value-container">
                <div id="prompt-value-inner">
                    <TextArea
                        value={promptString}
                        placeholder={t('promptPlaceholder')}
                        id="prompt-value-input"
                        onInput={(event) =>
                            setPromptString((event.target as any).value)
                        }
                        onKeyDown={(e) => onKeyup(e)}
                        disableKeyDown={waitingForResponse}
                    />

                    {showMicInitialization && (
                        <div className="pending-dots-container">
                            <Typography>{t('starting')}</Typography>
                        </div>
                    )}

                    {isProcessingSpeech && (
                        <div className="pending-dots-container">
                            <Typography>{t('processing')}</Typography>
                        </div>
                    )}

                    {isRecordingSpeech && (
                        <div className="recording-time-container">
                            <Typography>
                                {t('recordingStatus', {
                                    duration:
                                        '0:' +
                                        `00${Math.round(recordingTime)}`.slice(
                                            -2
                                        ),
                                })}
                            </Typography>
                        </div>
                    )}
                </div>
                <Stack
                    className="prompt-buttons"
                    display={'flex'}
                    direction={'row'}
                    alignItems={'center'}
                    justifyContent={'flex-end'}
                    spacing={themeSettings.space.ms}
                >
                    <Tooltip
                        title={
                            isRecordingSpeech
                                ? t('clickToStopRecording')
                                : t('clickForVoiceInput')
                        }
                    >
                        <span>
                            <Button
                                id="capture-audio-btn"
                                className="prompt-icon-button"
                                variant="outlined"
                                color="transparent"
                                onClick={onMicButtonClick}
                                disabled={disabled}
                            >
                                {!isRecordingSpeech && !isProcessingSpeech && (
                                    <IconElement
                                        type="microphone"
                                        size={themeSettings.font.sizes.lg}
                                    />
                                )}
                                {isRecordingSpeech && !isProcessingSpeech && (
                                    <IconElement
                                        type="shapes-shape-square"
                                        size={themeSettings.font.sizes.lg}
                                        fill={true}
                                    />
                                )}
                                {isProcessingSpeech && (
                                    <div className="mic-spinner">
                                        <CircularProgress
                                            size={
                                                themeSettings.iconSizes.medium
                                            }
                                        />
                                    </div>
                                )}
                            </Button>
                        </span>
                    </Tooltip>
                    <Tooltip title={disabled ? '' : t('send')}>
                        <span>
                            <Button
                                variant="contained"
                                className="prompt-icon-button"
                                onClick={() => onPrompt()}
                                disabled={
                                    promptString.trim() === '' || promptDisabled
                                }
                                data-is-disabled={disabled}
                            >
                                <IconElement
                                    type="send"
                                    size={themeSettings.font.sizes.lg}
                                    fill={true}
                                />
                            </Button>
                        </span>
                    </Tooltip>
                </Stack>
            </div>

            <Snackbar
                ContentProps={{
                    sx: {
                        backgroundColor: themeSettings.colors.n600,
                    },
                }}
                open={!!toastMessage}
                autoHideDuration={5000}
                onClose={handleToastClose}
                message={
                    <Typography variant="body2">{toastMessage}</Typography>
                }
            ></Snackbar>
        </Stack>
    )
}
