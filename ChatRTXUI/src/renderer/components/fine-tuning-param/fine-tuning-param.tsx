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

import './fine-tuning-param.scss'
import { Stack, TextField, Typography } from '@mui/material'
import React from 'react'
import { themeSettings } from '../../theme/theme'

export default function FineTuningParam({
    paramName,
    value,
    paramDetails,
    disabled,
}: {
    paramName: string
    value: number
    paramDetails: string
    disabled: boolean
}) {
    return (
        <div className="fine-tuning-param">
            <Stack
                direction={'column'}
                alignItems={'stretch'}
                justifyContent={'flex-start'}
                spacing={themeSettings.space.sm}
            >
                <Stack
                    direction={'row'}
                    alignItems={'center'}
                    justifyContent={'space-between'}
                >
                    <Typography variant="body1bold">{paramName}</Typography>
                    <TextField
                        sx={{
                            ['.MuiInputBase-root']: {
                                padding: '10px 4px 10px 4px !important',
                            },
                            width: '96px',
                        }}
                        variant="filled"
                        inputProps={{ type: 'number' }}
                        value={value}
                        disabled={disabled}
                    ></TextField>
                </Stack>
                <Stack className="param-desc">
                    <Typography variant="body1">{paramDetails}</Typography>
                </Stack>
            </Stack>
        </div>
    )
}
