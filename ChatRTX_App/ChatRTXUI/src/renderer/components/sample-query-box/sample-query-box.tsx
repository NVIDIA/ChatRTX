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

import './sample-query-box.scss'
import React, { useEffect, useState } from 'react'
import { DatasetSources, HistoryItem, ModelId } from '../../../electron/types'
import { Button, Stack, Typography } from '@mui/material'
import { ClientAPI } from '../../../electron/preload'
import { useTranslation } from 'react-i18next'
import OverflowTypography from '../overflow-typography/overflow-typography'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function SampleQueryBox() {
    const [activeModelId, setActiveModelId] = useState<ModelId>(
        clientAPI.getActiveModel()
    )

    const getSampleQuestions = () => {
        const modelId = clientAPI.getActiveModel()
        const datasetInfo = clientAPI.getDatasetInfo(modelId)
        const selected_path =
            datasetInfo.selected !== DatasetSources.DIRECTORY
                ? ''
                : datasetInfo.selected_path
        return clientAPI.getSampleQuestions(modelId, selected_path)
    }

    const { t } = useTranslation()
    const [history, setHistory] = useState<HistoryItem[]>(
        clientAPI.getHistory()
    )
    const [sampleQuestions, setSampleQuestions] =
        useState<string[]>(getSampleQuestions())

    useEffect(() => {
        const onHistoryUpdate = (data: HistoryItem[]) => setHistory(data)
        const historyListener = clientAPI.onHistoryUpdate(onHistoryUpdate)
        const datasetUpdateListener = clientAPI.onDatasetInfoUpdate(() => {
            setSampleQuestions(getSampleQuestions())
        })
        const activeModelUpdateListener = clientAPI.onActiveModelUpdate(() => {
            setActiveModelId(clientAPI.getActiveModel())
            setSampleQuestions(getSampleQuestions())
        })
        return () => {
            historyListener()
            datasetUpdateListener()
            activeModelUpdateListener()
        }
    }, [])

    const handleQueryClick = (query: string) => {
        clientAPI.sendPrompt(query, true)
    }

    return (
        <div className="sample-query-wrapper">
            {history.length === 0 && sampleQuestions.length !== 0 && (
                <Stack
                    flexDirection={'column'}
                    alignItems={'center'}
                    justifyContent={'center'}
                    className="sample-query-box"
                >
                    <div className="caption-text">
                        {activeModelId !== ModelId.Clip && (
                            <Typography variant="caption">
                                {t('sampleQuestionDescriptionDefault')}
                            </Typography>
                        )}
                        {activeModelId === ModelId.Clip && (
                            <Typography variant="caption">
                                {t('sampleQuestionDescriptionClip')}
                            </Typography>
                        )}
                    </div>
                    <div className="sample-question-container">
                        {sampleQuestions.map((query) => (
                            <div className="query-chip" key={query}>
                                <Button
                                    className="query-button"
                                    onClick={() => handleQueryClick(query)}
                                    color="transparent"
                                >
                                    <OverflowTypography
                                        variant="body1"
                                        className="query-text"
                                    >
                                        {query}
                                    </OverflowTypography>
                                </Button>
                            </div>
                        ))}
                    </div>
                </Stack>
            )}
        </div>
    )
}
