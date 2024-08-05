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

import {
    PaletteColor,
    SimplePaletteColorOptions,
    ThemeOptions,
    alpha,
    createTheme,
} from '@mui/material'
import React from 'react'

declare module '@mui/material/styles' {
    interface Palette {
        transparent: PaletteColor
        secondaryElevated: PaletteColor
        secondary700: PaletteColor
    }

    interface PaletteOptions {
        transparent: SimplePaletteColorOptions
        secondaryElevated: SimplePaletteColorOptions
        secondary700: SimplePaletteColorOptions
    }

    interface TypographyVariants {
        small: React.CSSProperties
        strong: React.CSSProperties
        dim: React.CSSProperties
        label: React.CSSProperties
        h6bold: React.CSSProperties
        body3: React.CSSProperties
        body1bold: React.CSSProperties
        body2bold: React.CSSProperties
        body3bold: React.CSSProperties
        body1italic: React.CSSProperties
        body2italic: React.CSSProperties
        body3italic: React.CSSProperties
    }

    // allow configuration using `createTheme`
    interface TypographyVariantsOptions {
        small?: React.CSSProperties
        strong?: React.CSSProperties
        dim?: React.CSSProperties
        label?: React.CSSProperties
        h6bold?: React.CSSProperties
        body3?: React.CSSProperties
        body1Medium?: React.CSSProperties
        body0bold?: React.CSSProperties
        body1bold?: React.CSSProperties
        body2bold?: React.CSSProperties
        body3bold?: React.CSSProperties
        body1italic?: React.CSSProperties
        body2italic?: React.CSSProperties
        body3italic?: React.CSSProperties
        caption0?: React.CSSProperties
    }
}

declare module '@mui/material/Button' {
    interface ButtonPropsColorOverrides {
        transparent: true
        secondaryElevated: true
        secondary700: true
    }
}

declare module '@mui/material/Typography' {
    interface TypographyPropsVariantOverrides {
        small: true
        strong: true
        dim: true
        label: true
        h6bold: true
        body3: true
        body0bold: true
        body1bold: true
        body2bold: true
        body3bold: true
        body1Medium: true
        body1italic: true
        body2italic: true
        body3italic: true
        caption0: true
    }
}

export const themeSettings = {
    iconSizes: {
        small: '0.5rem',
        medium: '1rem',
        large: '24px',
        regular: '20px',
    },
    inputs: {
        padding: '0',
        containerPadding: '0.601875em 1em',
    },
    buttons: {
        padding: '8px 16px',
    },
    font: {
        family: '"NVIDIA Sans", "NVIDIA Sans Fallback"',
        size: '16px',
        sizes: {
            xs: '10px',
            sm: '12px',
            ms: '14px',
            md: '16px',
            ml: '20px',
            lg: '24px',
            xl: '32px',
            xxl: '48px',
        },
        weights: {
            bold: '700',
            medium: '500',
            regular: '400',
            light: '300',
        },
    },
    space: {
        xxs: '2px',
        xs: '4px',
        sm: '8px',
        ms: '10px',
        md: '16px',
        ml: '24px',
        lg: '32px',
        xl: '48px',
        xxl: '64px',
    },
    colors: {
        brand: '#76b900',
        brandLight: '#8fcb2b',
        error: '#dc3528',
        success: '#76b900',
        warning: '#e7c32b',
        n000: '#ffffff',
        n025: '#fafafa',
        n050: '#afafaf',
        n100: '#a0a0a0',
        n200: '#919191',
        n300: '#828282',
        n400: '#737373',
        n500: '#636363',
        n600: '#494949',
        n650: '#393939',
        n700: '#323232',
        n800: '#292929',
        n900: '#191919',
        n950: '#111111',
        n1000: '#000000',
    },
    variables: {
        navBarHeight: '35px',
        appBarHeight: '64px',
        borderRadius: '6px',
    },
}

const styleId = 'chat-ui-vars'

const makeFromObj = (obj: { [key: string]: string }, key: string) =>
    Object.keys(obj).map((k) => `--${key}-${k}:${obj[`${k}`]};`)

if (!document.head.querySelector(`style#${styleId}`)) {
    const style = document.createElement('style')
    const styles = [`--fonts-nvidiaSans: ${themeSettings.font.family};`]
        .concat(makeFromObj(themeSettings.space, 'space'))
        .concat(makeFromObj(themeSettings.font.sizes, 'fontSizes'))
        .concat(makeFromObj(themeSettings.font.weights, 'fontWeights'))
        .concat(makeFromObj(themeSettings.colors, 'colors'))
        .concat(makeFromObj(themeSettings.variables, 'variables'))

    style.innerHTML = `:root {${styles.join('')}}`

    style.id = styleId

    document.head.appendChild(style)
}

const themeOptions: ThemeOptions = {
    typography: {
        fontFamily: 'NVIDIA',
        small: {
            fontSize: themeSettings.font.sizes.ms,
            opacity: 0.6,
        },
        strong: {
            fontWeight: 'bold',
        },
        dim: {
            opacity: 0.7,
        },
        label: {
            color: themeSettings.colors.n100,
            opacity: 0.75,
        },
        h1: {
            fontSize: '96px',
            fontWeight: '500',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '128px',
        },
        h2: {
            fontSize: '60px',
            fontWeight: '700',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '80px',
        },
        h3: {
            fontSize: '48px',
            fontWeight: '700',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '64px',
        },
        h4: {
            fontSize: '34px',
            fontWeight: '700',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '48px',
        },
        h5: {
            fontSize: '24px',
            fontWeight: '500',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '32px',
        },
        h6: {
            fontSize: '20px',
            fontWeight: '500',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '28px',
        },
        h6bold: {
            fontSize: '20px',
            fontWeight: '700',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '28px',
        },
        subtitle1: {
            fontSize: '16px',
            fontWeight: '500',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '24px',
        },
        subtitle2: {
            fontSize: '14px',
            fontWeight: '500',
            color: alpha('#FFFFFF', 0.9),
            lineHeight: '20px',
        },
        body1: {
            fontSize: '16px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.7),
            lineHeight: '24px',
        },
        body2: {
            fontSize: '14px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.7),
            lineHeight: '20px',
        },
        body3: {
            fontSize: '12px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.6),
            lineHeight: '16px',
        },
        body1Medium: {
            fontSize: '16px',
            fontWeight: '500',
            color: alpha('#FFFFFF', 0.7),
            lineHeight: '20px',
        },
        body0bold: {
            fontSize: '18px',
            fontWeight: '700',
            color: alpha('#FFFFFF', 0.7),
            lineHeight: '23px',
        },
        body1bold: {
            fontSize: '16px',
            fontWeight: '700',
            color: alpha('#FFFFFF', 0.7),
            lineHeight: '20px',
        },
        body2bold: {
            fontSize: '14px',
            fontWeight: '700',
            color: alpha('#FFFFFF', 0.7),
            lineHeight: '20px',
        },
        body3bold: {
            fontSize: '12px',
            fontWeight: '700',
            color: alpha('#FFFFFF', 0.7),
            lineHeight: '16px',
        },
        body1italic: {
            fontSize: '16px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.7),
            lineHeight: '24px',
            fontStyle: 'italic',
        },
        body2italic: {
            fontSize: '14px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.7),
            letterSpacing: '0.25px',
            fontStyle: 'italic',
        },
        body3italic: {
            fontSize: '12px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.7),
            letterSpacing: '0.25px',
            fontStyle: 'italic',
        },
        overline: {
            fontSize: '12px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.6),
            lineHeight: '20px',
            textTransform: 'uppercase',
        },
        caption0: {
            fontSize: '16px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.5),
            lineHeight: '20px',
        },
        caption: {
            fontSize: '14px',
            fontWeight: '400',
            color: alpha('#FFFFFF', 0.6),
            lineHeight: '20px',
        },
        button: {
            fontSize: '14px',
            fontWeight: '500',
            color: alpha('#FFFFFF', 1),
            lineHeight: '20px',
            textTransform: 'uppercase',
        },
    },
    palette: {
        transparent: {
            main: 'rgba(0,0,0,0)',
            light: themeSettings.colors.n600,
            dark: themeSettings.colors.n500,
            contrastText: themeSettings.colors.n000,
        },
        secondaryElevated: {
            main: themeSettings.colors.n800,
            light: themeSettings.colors.n900,
            dark: themeSettings.colors.n900,
            contrastText: themeSettings.colors.n000,
        },
        secondary700: {
            main: themeSettings.colors.n700,
            light: themeSettings.colors.n800,
            dark: themeSettings.colors.n800,
            contrastText: themeSettings.colors.n000,
        },
        primary: {
            main: themeSettings.colors.brand,
            light: themeSettings.colors.brand,
            dark: themeSettings.colors.brand,
        },
        secondary: {
            main: themeSettings.colors.n800,
            dark: themeSettings.colors.n700,
            contrastText: themeSettings.colors.n050,
        },
        background: {
            default: themeSettings.colors.n900,
            paper: themeSettings.colors.n700,
        },
        action: {
            disabledBackground: themeSettings.colors.n600,
            disabled: themeSettings.colors.n100,
        },
        text: {
            primary: themeSettings.colors.n050,
            secondary: themeSettings.colors.n600,
            disabled: themeSettings.colors.n300,
        },
    },

    components: {
        MuiInputBase: {
            styleOverrides: {
                root: {
                    fontSize: '14px',
                    fontWeight: '400',
                    lineHeight: '20px',
                    color: alpha('#FFFFFF', 0.75),
                },
                input: {
                    height: '20px',
                    paddingTop: '21px',
                    paddingBottom: '14px !important',
                    paddingLeft: '12px',
                    paddingRight: '12px',
                },
            },
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    borderBottom: '1px solid rgba(255, 255, 255, 0.30)',
                    backgroundColor: themeSettings.colors.n800,
                    ['.MuiFormLabel-root']: {
                        color: alpha('#FFFFFF', 0.3),
                        ['&.Mui-focused']: {
                            color: alpha('#FFFFFF', 0.3),
                        },
                        ['&[data-shrink="false"]']: {
                            transform: 'translate(12px, 21px) scale(1)',
                            fontSize: '14px',
                            fontWeight: '400',
                            lineHeight: '20px',
                            color: alpha('#FFFFFF', 0.6),
                        },
                        ['&[data-shrink="true"]']: {
                            transform: 'translate(12px, 3px) scale(1)',
                            fontSize: '12px',
                            fontWeight: '400',
                            lineHeight: '16px',
                        },
                    },
                },
            },
        },
        MuiOutlinedInput: {
            styleOverrides: {
                notchedOutline: {
                    border: '2px solid var(--colors-n600)',
                    borderRadius: 0,
                },
            },
        },
        MuiMenuItem: {
            styleOverrides: {
                root: {
                    padding: '14px 24px',
                    ['&:hover']: {
                        backgroundColor: alpha('#FFFFFF', 0.08),
                    },

                    ['&:active']: {
                        backgroundColor: alpha('#FFFFFF', 0.32),
                    },

                    ['&:focus-visible']: {
                        backgroundColor: alpha('#FFFFFF', 0.24),
                    },

                    ['&.Mui-selected']: {
                        backgroundColor: themeSettings.colors.n600,
                        ['&:hover, &:active, &:focus-visible']: {
                            backgroundColor: themeSettings.colors.n600,
                        },
                    },
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    borderRadius: 0,
                },
            },
        },
        MuiSelect: {
            styleOverrides: {
                root: {
                    borderRadius: `${themeSettings.variables.borderRadius}`,
                    ['.MuiOutlinedInput-notchedOutline']: {
                        border: 'none',
                    },
                    backgroundColor: themeSettings.colors.n800,
                },
                icon: {
                    color: alpha('#FFFFFF', 0.9),
                },
            },
        },
        MuiButton: {
            defaultProps: {
                disableFocusRipple: true,
            },

            styleOverrides: {
                root: {
                    minWidth: '0',
                    borderRadius: '6px',
                    boxShadow: 'unset',
                    textTransform: 'none',

                    ['&:hover, &:active, &:focus-visible']: {
                        boxShadow: 'unset',
                    },

                    ['&:hover span.MuiTouchRipple-root']: {
                        backgroundColor: alpha('#FFFFFF', 0.08),
                    },

                    ['&:active span.MuiTouchRipple-root']: {
                        backgroundColor: alpha('#FFFFFF', 0.32),
                    },

                    ['&:focus-visible span.MuiTouchRipple-root']: {
                        backgroundColor: alpha('#FFFFFF', 0.24),
                    },

                    ['&.MuiButton-colorTransparent']: {
                        color: '#FFFFFF',
                        opacity: 0.7,
                        ['&:hover, &:active, &:focus-visible']: {
                            opacity: 0.9,
                        },
                        ['&.Mui-disabled']: {
                            opacity: 0.38,
                        },
                    },
                },
                contained: {
                    fontSize: '14px',
                    fontWeight: '500',
                    lineHeight: '20px',
                    padding: themeSettings.buttons.padding,
                    textTransform: 'uppercase',
                    color: alpha('#FFFFFF', 1),
                    ['&.Mui-disabled']: {
                        color: alpha('#FFFFFF', 0.38),
                        backgroundColor: themeSettings.colors.n500,
                    },

                    ['&:hover span.MuiTouchRipple-root']: {
                        borderRadius: '6px',
                    },

                    ['&:active span.MuiTouchRipple-root']: {
                        borderRadius: '6px',
                    },

                    ['&:focus-visible span.MuiTouchRipple-root']: {
                        borderRadius: '6px',
                    },
                },
                outlined: {
                    fontSize: '14px',
                    fontWeight: '500',
                    lineHeight: '20px',
                    padding: themeSettings.buttons.padding,
                    textTransform: 'uppercase',
                    ['&.MuiButton-colorTransparent']: {
                        border: '1px solid rgba(255, 255, 255, 0.40)',
                        ['&:hover, &:active, &:focus-visible']: {
                            border: '1px solid rgba(255, 255, 255, 0.90)',
                        },
                        ['&.Mui-disabled']: {
                            border: '1px solid rgba(255, 255, 255, 0.38)',
                        },
                    },

                    ['&:hover span.MuiTouchRipple-root']: {
                        borderRadius: '6px',
                    },

                    ['&:active span.MuiTouchRipple-root']: {
                        borderRadius: '6px',
                    },

                    ['&:focus-visible span.MuiTouchRipple-root']: {
                        borderRadius: '6px',
                    },
                },
            },
        },
        MuiBackdrop: {
            styleOverrides: {
                root: {
                    width: '100%',
                    height: 'calc(100% - var(--variables-navBarHeight))',
                    top: themeSettings.variables.navBarHeight,
                    zIndex: '999999',
                },
            },
        },
        MuiModal: {
            styleOverrides: {
                root: {
                    width: '100%',
                    height: 'calc(100% - var(--variables-navBarHeight))',
                    top: themeSettings.variables.navBarHeight,
                    zIndex: '99999',
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    borderRadius: 1,
                },
            },
        },
        MuiAccordion: {
            styleOverrides: {
                root: {
                    boxShadow: 'none',
                    ['&.Mui-expanded']: {
                        margin: '0',
                    },
                    ['&.MuiPaper-root']: {
                        position: 'relative',
                    },
                },
            },
        },

        MuiAccordionSummary: {
            styleOverrides: {
                root: {
                    padding:
                        'var(--space-sm) var(--space-ms) var(--space-sm) var(--space-md)',
                    minHeight: 'unset',
                    ['&.Mui-expanded']: {
                        minHeight: 'unset',
                    },
                },
                content: {
                    margin: 0,
                    ['&.Mui-expanded']: {
                        margin: '0',
                    },
                },
            },
        },

        MuiAccordionDetails: {
            styleOverrides: {
                root: {
                    padding: 'var(--space-ms)',
                },
            },
        },
        MuiCheckbox: {
            styleOverrides: {
                root: {
                    color: 'rgba(255,255,255,0.38)',
                },
            },
        },
        MuiMenu: {
            styleOverrides: {
                paper: {
                    marginTop: '-35px',
                },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: themeSettings.colors.n650,
                    borderRadius: 0,
                    fontSize: '14px',
                    fontWeight: 400,
                    lineHeight: '20px',
                },
            },
        },
    },
}

const theme = createTheme(themeOptions)

export default theme
