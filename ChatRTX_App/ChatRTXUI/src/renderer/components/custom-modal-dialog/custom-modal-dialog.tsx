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

import './custom-modal-dialog.scss'
import React from 'react'
import { Button, Modal, Stack, Typography } from '@mui/material'
import { themeSettings } from '../../theme/theme'

interface customModalDialogProps {
    dialogHeader: string
    openDialog: boolean
    primaryButtonText?: string
    secondaryButtonText?: string
    onSecondaryButtonClick?: (
        event: React.MouseEvent<HTMLButtonElement, MouseEvent>
    ) => void
    onPrimaryButtonClick?: (
        event: React.MouseEvent<HTMLButtonElement, MouseEvent>
    ) => void
}

export default function CustomModalDialog({
    dialogHeader,
    openDialog,
    primaryButtonText,
    secondaryButtonText,
    onSecondaryButtonClick,
    onPrimaryButtonClick,
    children,
}: React.PropsWithChildren<customModalDialogProps>) {
    return (
        <Modal open={openDialog} className="custom-dialog">
            <Stack
                className="custom-dialog-wrapper"
                flexDirection={'column'}
                alignItems={'stretch'}
                justifyContent={'flex-start'}
                spacing={themeSettings.space.lg}
            >
                <Stack
                    className="dialog-content"
                    flexDirection={'column'}
                    alignItems={'stretch'}
                    justifyContent={'flex-start'}
                    spacing={'14px'}
                >
                    <Typography variant="h6">{dialogHeader}</Typography>
                    <div className="dialog-body">{children}</div>
                </Stack>
                <Stack
                    direction={'row'}
                    alignItems={'flex-end'}
                    justifyContent={'flex-end'}
                    spacing={themeSettings.space.ms}
                >
                    <Button
                        variant="outlined"
                        color="transparent"
                        onClick={onSecondaryButtonClick}
                    >
                        {secondaryButtonText}
                    </Button>
                    <Button onClick={onPrimaryButtonClick} variant="contained">
                        {primaryButtonText}
                    </Button>
                </Stack>
            </Stack>
        </Modal>
    )
}
