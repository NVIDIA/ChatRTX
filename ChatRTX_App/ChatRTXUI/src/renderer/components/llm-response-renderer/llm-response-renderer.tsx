/* SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import React from 'react'
import Markdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import { PrismAsync as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import CustomIconButton from '../custom-icon-button/custom-icon-button'
import { themeSettings } from '../../theme/theme'
import { useTranslation } from 'react-i18next'
import { Typography } from '@mui/material'

export const toText = (node: React.ReactNode): string => {
    if (
        typeof node === 'string' ||
        typeof node === 'number' ||
        typeof node === 'boolean'
    ) {
        return node.toString()
    }
    if (!node) {
        return ''
    }
    if (Array.isArray(node)) {
        return node.map((entry) => toText(entry)).join('')
    }
    // Because ReactNode includes {} in its union we need to jump through a few hoops.
    const props: { children?: React.ReactNode } = (node as any).props
        ? (node as any).props
        : {}

    if (!props || !props.children) {
        return ''
    }

    return toText(props.children)
}

const CodeBlock = ({
    codeText,
    language,
}: {
    language: string
    codeText: string
}) => {
    // console.log('Logging children ', codeText)

    const { t } = useTranslation()

    return (
        <div
            style={{
                position: 'relative',
            }}
        >
            <div
                style={{
                    position: 'absolute',
                    top: '8px',
                    right: '8px',
                }}
            >
                <CustomIconButton
                    size={themeSettings.iconSizes.medium}
                    type="copy-clipboard"
                    tooltip={t('copyCode')}
                    onClick={() => {
                        const tempDivElement = document.createElement('div')
                        tempDivElement.innerHTML = codeText
                        navigator.clipboard.writeText(
                            tempDivElement.textContent ||
                                tempDivElement.innerText ||
                                ''
                        )
                    }}
                    className="copy-clipboard-code"
                ></CustomIconButton>
            </div>

            <SyntaxHighlighter
                language={language ?? 'bash'}
                style={vscDarkPlus}
            >
                {codeText}
            </SyntaxHighlighter>
        </div>
    )
}

export const isMultilineCodeBlock = (children: React.ReactNode): boolean => {
    if (typeof children === 'string') return children.includes('\n')
    return isMultilineCodeBlock(toText(children))
}

const Code: React.FC<{ codeText: string; language: string }> = ({
    codeText,
    language,
}) => {
    if (isMultilineCodeBlock(codeText)) {
        return <CodeBlock codeText={codeText} language={language} />
    }

    return <code>{codeText}</code>
}

const RichResponse = ({ response }: { response: string }) => {
    return (
        <Markdown
            rehypePlugins={[rehypeRaw]}
            components={{
                img(props) {
                    return (
                        <img
                            src={String(
                                props.src ?? props.node.properties['dataLink']
                            )}
                        />
                    )
                },
                code(props) {
                    const { children, node, className } = props
                    //   console.log('logging props ', className)
                    const lang = className
                        ?.substring(className.indexOf('language-') + 9)
                        ?.split(',')[0]
                    // REVIEW: is this check necessary?
                    if (node?.tagName === 'code') {
                        return (
                            <Code codeText={toText(children)} language={lang} />
                        )
                    }

                    return <>{children}</>
                },
                h1(props) {
                    return (
                        <Typography variant="h5">{props.children}</Typography>
                    )
                },
                h2(props) {
                    return (
                        <Typography variant="h6">{props.children}</Typography>
                    )
                },
                h3(props) {
                    return (
                        <Typography variant="subtitle1">
                            {props.children}
                        </Typography>
                    )
                },
                h4(props) {
                    return (
                        <Typography variant="subtitle1">
                            {props.children}
                        </Typography>
                    )
                },
                h5(props) {
                    return (
                        <Typography variant="subtitle2">
                            {props.children}
                        </Typography>
                    )
                },
                h6(props) {
                    return (
                        <Typography variant="subtitle2">
                            {props.children}
                        </Typography>
                    )
                },
            }}
        >
            {response}
        </Markdown>
    )
}

const RenderResponse = ({ response }: { response: any[] }) => {
    return (
        <div>
            {response.map((item: any, index: number) => {
                switch (item.type) {
                    case 'text':
                        return <p key={index}>{item.content}</p>
                    case 'image':
                        return (
                            <img
                                key={index}
                                src={item.url}
                                alt={item.alt || 'Image'}
                            />
                        )
                    case 'code':
                        return <code key={index}>{item.content}</code>
                    default:
                        return <p key={index}>Unsupported content</p>
                }
            })}
        </div>
    )
}

export const LLMResponseRenderer = ({ response }: { response: any }) => {
    if (typeof response === 'string') {
        return <RichResponse response={response} />
    }

    if (Array.isArray(response)) {
        return <RenderResponse response={response} />
    }

    return <p>Unsupported response format</p>
}
