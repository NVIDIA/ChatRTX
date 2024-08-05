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

import './text-area.scss'
import React, {
    FormEvent,
    KeyboardEvent,
    useEffect,
    useRef,
    useState,
} from 'react'

export default function TextArea({
    value,
    disabled,
    placeholder,
    id,
    disableKeyDown,
    onInput,
    onKeyDown,
}: {
    value: string
    disabled?: boolean
    placeholder?: string
    id?: string
    disableKeyDown?: boolean
    onInput?: (event: FormEvent<HTMLTextAreaElement>) => void
    onKeyDown?: (event: KeyboardEvent<HTMLTextAreaElement>) => void
}) {
    const textareaRef = useRef<HTMLTextAreaElement>()
    const [focused, setFocused] = useState<boolean>(false)
    const [customResized, setCustomResized] = useState<boolean>(false)

    useEffect(() => {
        if (!textareaRef.current) {
            return
        }

        const observer = new MutationObserver((mutations) => {
            if (
                !textareaRef.current ||
                !mutations ||
                !mutations[0] ||
                mutations[0].attributeName !== 'style'
            ) {
                return
            }
            setCustomResized(parseInt(textareaRef.current.style.height) > 0)
        })

        observer.observe(textareaRef.current, {
            attributes: true,
            attributeFilter: ['style'],
        })

        return () => observer.disconnect()
    }, [textareaRef.current])

    const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
        if (
            event.key === 'Enter' &&
            !event.ctrlKey &&
            !event.shiftKey &&
            !event.altKey &&
            !!textareaRef.current &&
            textareaRef.current.value.trim()
        ) {
            textareaRef.current.blur()
        }

        if (typeof onKeyDown === 'function') {
            onKeyDown(event)
        }
    }

    return (
        <div
            className="text-area-container"
            id={id}
            data-disabled={disabled}
            data-focused={focused}
            data-value={value}
            data-has-value={!!value.trim()}
            data-custom-resized={customResized}
        >
            <textarea
                ref={textareaRef}
                value={value}
                disabled={disabled}
                placeholder={placeholder}
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
                onInput={(event) =>
                    typeof onInput === 'function' ? onInput(event) : undefined
                }
                onKeyDown={(event) =>
                    !disableKeyDown ? handleKeyDown(event) : () => {}
                }
                aria-multiline="true"
                rows={1}
            ></textarea>
        </div>
    )
}
