# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import gradio as gr
import subprocess
from ui.configuration import Configuration
import uuid
import sys
import os
import kaizen_theme as kaizen
import re
import threading
import webbrowser
import socket
import random
import copy
import asyncio

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class MainInterface:

    _dataset_path_key = 'dataset'
    _models_path_key = 'models'
    _dataset_directory_label_text = ".txt, .pdf, .doc, .docx files supported"
    _data_directory_clip_label_text = ".jpeg, .png, .bmp files supported"
    _dataset_path_updated_callback = None
    _dataset_source_updated_callback = None
    _shutdown_callback = None
    _reset_chat_callback = None
    _undo_last_chat_callback = None
    _model_change_callback = None
    _regenerate_index_callback = None
    _on_mic_button_click_callback = None
    _on_mic_recording_done_callback = None
    _on_model_downloaded_callback = None
    _on_model_installed_callback = None
    _on_model_delete_callback = None
    _query_handler = None
    _state = None
    _interface = None
    _streaming = False
    _models_info_map = {}
    _is_asr_model_loaded = False
    _asr_model_state = None
    _installed_models = []
    _available_to_download_models = []
    _installed_dropdown_selected_model = None
    _download_dropdown_selected_model = None
    _ready_to_install_model = None
    _chinese_model_name = "ChatGLM 3 6B int4 (Supports Chinese)"
    _clip_model_name = "CLIP"
    _root_path = os.getcwd()
    _allowed_paths = [os.path.join(_root_path, 'Temp')]
    _model_uninstall_progress_counter = 0
    _https_enabled = os.path.exists('certs/servercert.pem') and os.path.exists('certs/serverkey.pem')
    _is_asr_enabled = False

    def _get_enable_disable_elemet_list(self):
        ret_val = [
            self._chat_query_input_textbox,
            self._chat_bot_window,
            self._chat_submit_button,
            self._chat_retry_button,
            self._chat_reset_button,
            self._model_dropdown,
            self._dataset_source_dropdown,
            self._dataset_update_source_edit_button,
            self._dataset_regenerate_index_button,
            self._settings_button
        ]

        if self._is_asr_enabled:
            ret_val.append(self._mic_start_button)

        if self._chat_undo_button is not None:
            ret_val.append(self._chat_undo_button)
        
        return ret_val
    
    def _before_change_element_state(self, request: gr.Request):
        self._validate_session(request)
        ret_val = [
            gr.Textbox("", interactive=False),
            gr.Chatbot(visible=True),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Dropdown(interactive=False),
            gr.Dropdown(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False)
        ]

        if self._is_asr_enabled:
            ret_val.append(gr.Button(interactive=False))

        if self._chat_undo_button is not None:
            ret_val.append(gr.Button(interactive=False))

        return ret_val + self._get_sample_question_components_new(True)
    
    def _after_change_element_state(self, request: gr.Request):
        self._validate_session(request)
        ret_val = [
            gr.Textbox(interactive=True),
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Dropdown(interactive=True),
            gr.Dropdown(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True)
        ]

        if self._is_asr_enabled:
            ret_val.append(gr.Button(interactive=True))

        if self._chat_undo_button is not None:
            ret_val.append(gr.Button(interactive=True))

        return ret_val
    
    def _element_list_when_response_under_process(self):
        ret_val = [
            self._chat_query_input_textbox,
            self._chat_submit_button,
            self._chat_retry_button,
            self._chat_reset_button,
            self._model_dropdown,
            self._dataset_source_dropdown,
            self._dataset_update_source_edit_button,
            self._dataset_regenerate_index_button,
            self._settings_button
        ]

        for sample in self._sample_question_components:
            ret_val.append(sample['component'])

        if self._is_asr_enabled:
            ret_val.append(self._mic_start_button)

        if self._chat_undo_button is not None:
            ret_val.append(self._chat_undo_button)

        return ret_val
    
    def _before_state_when_response_processing(self, request: gr.Request):
        self._validate_session(request)
        ret_val = [
            gr.Textbox(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Dropdown(interactive=False),
            gr.Dropdown(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False)
        ] + [gr.Button(interactive=False)] * len(self._sample_question_components)

        if self._is_asr_enabled:
            ret_val.append(gr.Button(interactive=False))

        if self._chat_undo_button is not None:
            ret_val.append(gr.Button(interactive=False))
        return ret_val 
    
    def _after_state_when_response_processing(self, request: gr.Request):
        self._validate_session(request)
        ret_val = [
            gr.Textbox(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Dropdown(interactive=True),
            gr.Dropdown(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True)
        ] + [gr.Button(interactive=True)] * len(self._sample_question_components)

        if self._is_asr_enabled:
            ret_val.append(gr.Button(interactive=True))

        if self._chat_undo_button is not None:
            ret_val.append(gr.Button(interactive=True))

        return ret_val

    def __init__(self, chatbot=None, streaming = False) -> None:
        self._interface = None
        self._query_handler = chatbot
        self._streaming = streaming
        self.config = Configuration()
        self._dataset_path = self._get_dataset_path()
        self._default_dataset_path = self._get_default_dataset_path()
        self._default_chinese_dataset_path = self._get_default_chinese_dataset_path()
        self._default_clip_dataset_path = self._get_default_clip_dataset_path()
        pass

    def _get_dataset_path(self):
        _dataset_path = ""
        dataset_config = self.config.get_config(self._dataset_path_key) or {}
        if 'path' in dataset_config:
            _dataset_path = dataset_config['path']
            if 'isRelative' in dataset_config and dataset_config['isRelative'] is True:
                _dataset_path = os.path.join(os.getcwd(), _dataset_path)
        
        return _dataset_path
    
    def _get_default_dataset_path(self):
        _dataset_path = ""
        dataset_config = self.config.get_config_from_file(self._dataset_path_key, "config/config.json") or {}
        if 'path' in dataset_config:
            _dataset_path = dataset_config['path']
            if 'isRelative' in dataset_config and dataset_config['isRelative'] is True:
                _dataset_path = os.path.join(os.getcwd(), _dataset_path)
        
        return _dataset_path
    
    def _get_default_chinese_dataset_path(self):
        _dataset_path = ""
        dataset_config = self.config.get_config_from_file(self._dataset_path_key, "config/config.json") or {}
        if 'path' in dataset_config:
            _dataset_path = dataset_config['path_chinese']
            if 'isRelative' in dataset_config and dataset_config['isRelative'] is True:
                _dataset_path = os.path.join(os.getcwd(), _dataset_path)
        
        return _dataset_path
    
    def _get_default_clip_dataset_path(self):
        _dataset_path = ""
        dataset_config = self.config.get_config_from_file(self._dataset_path_key, "config/config.json") or {}
        if 'path' in dataset_config:
            _dataset_path = dataset_config['path_clip']
            if 'isRelative' in dataset_config and dataset_config['isRelative'] is True:
                _dataset_path = os.path.join(os.getcwd(), _dataset_path)
        
        return _dataset_path

    def on_dataset_path_updated(self, callback):
        self._dataset_path_updated_callback = callback

    def on_dataset_source_updated(self, callback):
        self._dataset_source_updated_callback = callback

    def on_shutdown(self, callback):
        self._shutdown_callback = callback

    def on_reset_chat(self, callback):
        self._reset_chat_callback = callback

    def on_undo_last_chat(self, callback):
        self._undo_last_chat_callback = callback

    def on_model_change(self, callback):
        self._model_change_callback = callback

    def on_regenerate_index(self, callback):
        self._regenerate_index_callback = callback

    def on_mic_button_click(self, callback):
        self._on_mic_button_click_callback = callback

    def on_mic_recording_done(self, callback):
        self._on_mic_recording_done_callback = callback

    def on_model_downloaded(self, callback):
        self._on_model_downloaded_callback = callback

    def on_model_installed(self, callback):
        self._on_model_installed_callback = callback

    def on_model_delete(self, callback):
        self._on_model_delete_callback = callback

    def _get_theme(self):
        primary_hue = gr.themes.Color("#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900")
        neutral_hue = gr.themes.Color("#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929")
        return gr.Theme(
            primary_hue=primary_hue,
            neutral_hue=neutral_hue
        ).set(
            body_background_fill="#191919",
            body_background_fill_dark="#191919",
            block_background_fill="#292929",
            block_background_fill_dark="#292929",
            block_label_background_fill="#292929",
            block_label_background_fill_dark="#292929",
            border_color_primary="#191919",#components background
            border_color_primary_dark="#191919",
            background_fill_primary="#292929",#dropdown
            background_fill_primary_dark="#292929",
            background_fill_secondary="#393939",#response chatbot bubble
            background_fill_secondary_dark="#393939",
            color_accent_soft="#393939",#request chatbot bubble
            color_accent_soft_dark="#393939",
            #text colors
            block_label_text_color="#FFFFFF",
            block_label_text_color_dark="#FFFFFF",
            body_text_color="#FFFFFF",
            body_text_color_dark="#FFFFFF",
            body_text_color_subdued="#FFFFFF",
            body_text_color_subdued_dark="#FFFFFF",
            button_secondary_text_color="#FFFFFF",
            button_secondary_text_color_dark="#FFFFFF",
            button_primary_text_color="#FFFFFF",
            button_primary_text_color_dark="#FFFFFF",
            input_placeholder_color="#FFFFFF",#placeholder text color
            input_placeholder_color_dark="#FFFFFF",
        )

    def get_css(self):
        return kaizen.css() + open(os.path.join(os.path.dirname(__file__), 'www/app.css')).read()

    def render(self):
        with gr.Blocks(
            title="ChatRTX",
            analytics_enabled=False,
            theme=kaizen.theme(),
            css=self.get_css(),
            js=os.path.join(os.path.dirname(__file__), 'www/app.js')
        ) as interface:
            self._interface = interface
            self.update_model_info()
            self._state = gr.State({})
            self._asr_model_state = gr.HTML("on", elem_id="asr_model_state", visible=False)
            self._hidden_checkbox = gr.Checkbox(visible=False, interactive=True)
            self._is_asr_enabled = self.config.get_config('models/enable_asr')
            (
                self._settings_button,
                self._shutdown_button,
                self._shutdown_post_shutdown_group,
                self._shutdown_memory_released_markdown,
                self._shutdown_invalid_session_markdown
            ) = self._render_logo_shut_down()
            (
                self._install_progress_group,
                self._install_progress_markdown
            ) = self._render_install_progress()
            with gr.Column(visible=False) as self._settings_page:
                (
                    self._settings_page_group,
                    self._settings_close_button,
                    self._uninstall_model_dict
                ) = self._render_settings_page()
            with gr.Column() as self._chat_page:
                with gr.Row():
                    (
                        self._model_dropdown,
                        self._model_group,
                        self._add_model_accordion,
                        self._install_model_group,
                        self._download_model_group,
                        self._cancel_install_btn,
                        self._install_model_btn,
                        self._model_name_markdown,
                        self._model_info_markdown,
                        self._download_model_dropdown,
                        self._download_model_button,
                        self._model_license_row,
                        self._model_license_markdown,
                        self._license_agreement
                    ) = self._render_models()
                    (
                        self._dataset_source_textbox,
                        self._dataset_update_source_edit_button,
                        self._dataset_source_dropdown,
                        self._dataset_regenerate_index_button,
                        self._dataset_label_markdown,
                        self._dataset_group
                    ) = self._render_dataset_picker()
                (
                    self._sample_question_components,
                    self._sample_question_rows,
                    self._sample_question_empty_space_component,
                    self._sample_qustion_default_dataset_markdown,
                    self._sample_question_component_group,
                    self._english_sample_questions,
                    self._chinese_sample_questions,
                    self._clip_sample_questions,
                ) = self._render_sample_question()
                (
                    self._chat_bot_window,
                    self._chat_query_input_textbox,
                    self._chat_mic_component,
                    self._mic_start_button,
                    self._mic_stop_button,
                    self._chat_submit_button,
                    self._chat_retry_button,
                    self._chat_undo_button,
                    self._chat_reset_button,
                    self._chat_query_group,
                    self._chat_disclaimer_markdown,
                    self._chat_action_buttons_row
                ) = self._render_chatbot(show_chatbot=len(self._sample_question_components) == 0)
            self._handle_events()
            self._handle_links()
        interface.queue()
        port = self._get_free_port()
        self._open_app(port)
        if (self._https_enabled):
            interface.launch(
                favicon_path=os.path.join(os.path.dirname(__file__), 'assets/nvidia_logo.png'),
                show_api=False,
                server_port=port,
                allowed_paths=['Temp/Temp_Images/.', 'Temp/.'],
                ssl_certfile='certs/servercert.pem',
                ssl_keyfile='certs/serverkey.pem'
            )
        else:
            interface.launch(
                favicon_path=os.path.join(os.path.dirname(__file__), 'assets/nvidia_logo.png'),
                show_api=False,
                server_port=port,
                allowed_paths=['Temp/Temp_Images/.', 'Temp/.']
            )

    def _get_free_port(self):
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set a short timeout for the connection attempt
        sock.settimeout(1)
        port = None
        while port is None:
            port = random.randint(1024, 49000)
            try:
                # Attempt to bind to the port
                sock.bind(("127.0.0.1", port))
            except OSError as e:
                port = None
                if e.errno != 98:  # errno 98: Address already in use
                    print('OS error', e)
                    break 
        sock.close()
        return port

    def _open_app(self, port):
        def launch_thread(cookie):
            if self._https_enabled:
                launch_url = f'https://127.0.0.1:{port}?cookie={cookie}&__theme=dark'
            else:
                launch_url = f'http://127.0.0.1:{port}?cookie={cookie}&__theme=dark'
            print(f'Open {launch_url} in browser to start ChatRTX')
            webbrowser.open(launch_url)
            return None
        
        self._secure_cookie = str(uuid.uuid4())
        threading.Thread(target=launch_thread, args=(self._secure_cookie,)).start()
        return None

    def _validate_request(self, request: gr.Request):
        headers = request.headers
        session_key = None
        if 'cookie' in headers:
            cookies = headers['cookie']
            if '_s_chat_=' in cookies:
                cookies = cookies.split('; ')
                for i, cookie in enumerate(cookies):
                    key, value = cookie.split('=')
                    if key == '_s_chat_':
                        session_key = value
        
        if session_key == None or session_key != self._secure_cookie:
            raise 'session validation failed'
        
        return True

    def _handle_links(self):
        def open_local_link(local_path, request: gr.Request):
            self._validate_session(request)
            path = os.path.join(os.getcwd(), local_path)

            should_go_to_path = False
            for allowed_path in self._allowed_paths:
                if (os.path.exists(path) and path.startswith(allowed_path)):
                    should_go_to_path = True
                    break

            if (not should_go_to_path):
                return None
            subprocess.Popen([sys.executable, "./ui/open_local_path.py", path])
            return None

        file_path = gr.Textbox(visible=False)
        gr.Button(visible=False).click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            open_local_link,
            file_path,
            None,
            api_name="open_local_link"
        )

    def _get_session_id(self, state):
        if isinstance(state, object):
            if not 'session-id' in state:
                state['session-id'] = str(uuid.uuid4())
            return state['session-id']
        return None
    
    def _render_settings_page(self):
        uninstall_model_dict = {}
        with gr.Group(elem_classes="padding-8p settings-page-group") as settings_page_group:
            with gr.Row(elem_classes="settings-page-title-row padding-8p"):
                gr.Markdown("Settings", elem_classes="settings-page-title")
                settings_close_button = gr.Button(
                "",
                scale=0,
                icon=os.path.join(os.path.dirname(__file__), 'assets/close.png'),
                elem_classes="icon-button tooltip-component transparent-background-button",
                elem_id="settings-close-btn"
            )
            with gr.Group(elem_classes="padding-8p") as ai_model_management_group:
                gr.Markdown("AI model management", elem_classes="font-bold")
                _id = 0
                for model_key in self._models_info_map:
                    isInstalled = model_key in self._installed_models   
                    with gr.Row(elem_classes="uninstall-model-row", visible=isInstalled) as uninstall_model_row:
                        isActiveModel = model_key == self._installed_dropdown_selected_model
                        model_name = gr.HTML(model_key, elem_classes="uninstall-model-name", visible=not isActiveModel)
                        model_name_with_active_tag = gr.HTML(model_key + '<b> | Active</b>', visible=isActiveModel, elem_classes="uninstall-model-name")
                        gr.HTML(self._models_info_map[model_key]["model_size"], elem_classes="uninstall-model-size")
                        uninstall_button = gr.Button(
                            "",
                            elem_classes="tooltip-component icon-button transparent-background-button",
                            scale=0,
                            icon=os.path.join(os.path.dirname(__file__), 'assets/reset.png'),
                            interactive= not isActiveModel,
                            elem_id="delete-model-btn-"+str(_id)
                        )
                        progress_spinner_button = gr.Button(
                            "",
                            elem_classes="icon-button transparent-background-button-2",
                            scale=0,
                            visible=False,
                            icon=os.path.join(os.path.dirname(__file__), "assets/animate_loading.gif"),
                            elem_id="delete-model-progress-btn-"+str(_id)
                        )
                    uninstall_model_dict[model_key] = {
                        'model_row': uninstall_model_row,
                        'model_name_html': model_name,
                        'delete_button': uninstall_button,
                        'progress_spinner_button': progress_spinner_button,
                        'model_name_with_active_tag_html': model_name_with_active_tag
                    }
                    _id+=1

        return settings_page_group, settings_close_button, uninstall_model_dict

    def _render_install_progress(self):
        with gr.Group(visible=False, elem_classes="install-progress-group") as install_progress_group:
            with gr.Row():
                gr.HTML("")
                # Chnage below asset to loading spinner
                gr.Image(os.path.join(os.path.dirname(__file__), "assets/animate_loading.gif"),
                    interactive=False,
                    show_label=False,
                    show_download_button=False,
                    width=40,
                    scale=0,
                    container=False,
                    min_width=40
                )
                gr.HTML("")
            with gr.Column():
                # Fill the markdown as per current model that is installing
                install_progress_markdown = gr.Markdown(
                    "",
                    elem_classes="text-align-center model-name-install-progress"
                )
                gr.Markdown(
                    "This may take a few minutes...",
                    elem_classes="text-align-center intall-time-progress-text"
                )
        return (install_progress_group,
                    install_progress_markdown)


    def _render_models(self):
        with gr.Column():
            with gr.Group(elem_classes="padding-8p model-goup") as model_group:
                gr.Markdown("<b>AI model</b>")
                gr.Markdown(
                    'Select AI model',
                    elem_classes="description-secondary-markdown"
                )
                model_dropdown = gr.Dropdown(
                    self._installed_models,
                    value= self._installed_dropdown_selected_model,
                    elem_classes="height-40p margin-bottom-8p ui-dropdown",
                    container=False,
                    filterable = False
                )
                with gr.Accordion(label="Add new models", elem_classes="padding-8p add-model-accordion", open=False, visible = (self._ready_to_install_model is not None or self._download_dropdown_selected_model is not None)) as add_model_accordion:
                    with gr.Group(visible = self._ready_to_install_model != None, elem_classes="install-model-group") as install_model_group:
                        with gr.Column():
                            with gr.Row():
                                if self._ready_to_install_model:
                                    model_name_markdown = gr.Markdown(self._ready_to_install_model, elem_classes="model-name-markdown")
                                else:
                                    model_name_markdown = gr.Markdown(elem_classes="model-name-markdown")
                                cancel_install_btn = gr.Button(
                                    "Cancel",
                                    variant="secondary",
                                    scale=0,
                                    visible= (self._ready_to_install_model is not None),
                                    min_width=100
                                )
                                install_model_btn =  gr.Button(
                                    "INSTALL",
                                    variant="primary",
                                    scale=0,
                                    visible= (self._ready_to_install_model is not None),
                                    min_width=100
                                )
                    with gr.Group(visible = self._ready_to_install_model == None) as download_model_group:
                        with gr.Row():
                            download_model_dropdown = gr.Dropdown(
                                self.get_formatted_model_name_with_size(self._available_to_download_models),
                                value=self.get_formatted_model_name_with_size(self._download_dropdown_selected_model),
                                container=False,
                                elem_classes="height-40p ui-dropdown",
                                filterable=False
                            )
                            download_model_button = gr.Button(
                                "",
                                scale=0,
                                icon=os.path.join(os.path.dirname(__file__), 'assets/download.png'),
                                elem_classes="icon-button tooltip-component",
                                elem_id="download_model_button"
                            )
                    if self._ready_to_install_model:
                        model_info_markdown = gr.Markdown(self._models_info_map[self._ready_to_install_model]["model_info"], elem_classes="model-info-text")
                    elif self._download_dropdown_selected_model:
                        model_info_markdown = gr.Markdown(self._models_info_map[self._download_dropdown_selected_model]["model_info"], elem_classes="model-info-text")
                    else:
                        model_info_markdown = gr.Markdown(elem_classes="model-info-text")
                    with gr.Row(visible=self._download_dropdown_selected_model is not None and self._ready_to_install_model is None) as model_license_row:
                        license_agreement = gr.Checkbox(value=False, min_width=40, label="", scale=0, elem_classes="agreement-checkbox transparent-background-button-2")
                        if self._ready_to_install_model:
                            model_license_markdown = gr.Markdown(self._models_info_map[self._ready_to_install_model]["model_license"], elem_classes="model-info-text")
                        elif self._download_dropdown_selected_model:
                            model_license_markdown = gr.Markdown(self._models_info_map[self._download_dropdown_selected_model]["model_license"], elem_classes="model-info-text")
                        else:
                            model_license_markdown = gr.Markdown(elem_classes="model-info-text")

        return (model_dropdown, model_group, add_model_accordion, install_model_group, download_model_group, cancel_install_btn, install_model_btn, model_name_markdown, model_info_markdown, download_model_dropdown, download_model_button, model_license_row, model_license_markdown ,license_agreement)


    def _render_logo_shut_down(self):
        with gr.Row():
            gr.Image(os.path.join(os.path.dirname(__file__), "assets/nvidia_logo.png"),
                interactive=False,
                show_label=False,
                show_download_button=False,
                width=40,
                scale=0,
                container=False,
                min_width=40
            )
            gr.HTML("""
                <h1 style="font-size:32px; line-height:40px; margin:0; padding:0">ChatRTX</h1>
            """)
            settings_button = gr.Button(
                "",
                scale=0,
                icon=os.path.join(os.path.dirname(__file__), 'assets/settings.png'),
                elem_classes="icon-button tooltip-component",
                elem_id="settings-btn"
            )
            shutdown_button = gr.Button(
                "",
                scale=0,
                icon=os.path.join(os.path.dirname(__file__), 'assets/shutdown.png'),
                elem_classes="icon-button tooltip-component",
                elem_id="shutdown-btn"
            )
        
        with gr.Group(visible=False, elem_classes="shutdown-group") as post_shutdown_group:
            with gr.Row():
                gr.HTML("")
                gr.Image(os.path.join(os.path.dirname(__file__), "assets/info.png"),
                    interactive=False,
                    show_label=False,
                    show_download_button=False,
                    width=40,
                    scale=0,
                    container=False,
                    min_width=40
                )
                gr.HTML("")
            with gr.Row():
                shutdown_memory_released_markdown = gr.Markdown(
                    "Video memory released. Reopen ChatRTX from desktop to continue chatting.",
                    elem_classes="text-align-center"
                )
                shutdown_invalid_session_markdown = gr.Markdown(
                    "Invalid session. Reopen ChatRTX from desktop to continue chatting.",
                    elem_classes="text-align-center"
                )

        return settings_button, shutdown_button, post_shutdown_group, shutdown_memory_released_markdown, shutdown_invalid_session_markdown

    def _render_dataset_picker(self):
        sources = copy.deepcopy(self.config.get_config("dataset/sources"))
        self._dataset_selected_source = "directory"
        self._selected_model = self.config.get_config("models/selected")
        if self._selected_model == self._clip_model_name:
            sources.remove("nodataset")
            if (self._dataset_selected_source == "nodataset"):
                self._dataset_selected_source = "directory"
        self.config.set_config("dataset/selected", self._dataset_selected_source)

        with gr.Column(elem_classes="dataset-column"):
            with gr.Group(elem_classes="padding-8p dataset-goup") as dataset_group:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("<b>Dataset</b>")
                        dataset_label_markdown = gr.Markdown(
                            self._dataset_directory_label_text if self._selected_model != self._clip_model_name else self._data_directory_clip_label_text,
                            elem_classes="description-secondary-markdown",
                            elem_id="dataset-description-label"
                        )
                    regenerate_vector_button = gr.Button(
                        "", 
                        icon=os.path.join(os.path.dirname(__file__), 'assets/regenerate.png'),
                        elem_classes="icon-button tooltip-component",
                        elem_id="dataset-regenerate-index-btn",
                        scale=0
                    )
                dataset_source_dropdown = gr.Dropdown(
                    self.config.get_display_strings(sources),
                    value=lambda: self.config.get_display_strings(self._dataset_selected_source),
                    show_label=False,
                    container=False,
                    elem_classes="margin-bottom-8p height-40p ui-dropdown",
                    filterable = False
                )
                with gr.Row():
                    dataset_source_textbox = gr.Textbox(
                        lambda: self._dataset_path,
                        scale=9,
                        container=False,
                        elem_classes="height-40p margin-right-8p",
                        interactive=False,
                        placeholder="Enter URL...",
                        max_lines=1,
                        autoscroll=True,
                        visible=self._dataset_selected_source=="directory",
                    )
                    dataset_update_source_edit_button = gr.Button(
                        "",
                        icon=os.path.join(os.path.dirname(__file__), 'assets/edit.png'),
                        elem_classes="icon-button tooltip-component",
                        elem_id="dataset-update-source-edit-button",
                        visible=self._dataset_selected_source=="directory",
                        scale=0
                    )
                    return (
                        dataset_source_textbox,
                        dataset_update_source_edit_button,
                        dataset_source_dropdown,
                        regenerate_vector_button,
                        dataset_label_markdown,
                        dataset_group
                    )


    def _render_sample_question(self):
        question_butons = []
        question_rows = []
        sample_questions: list = self.config.get_config("sample_questions")
        sample_questions_chinese: list = self.config.get_config("sample_questions_chinese")
        sample_questions_clip: list = self.config.get_config("sample_questions_clip")
        if sample_questions is None or len(sample_questions) == 0:
            return question_butons, question_rows
        
        elem_per_row = 2
        with gr.Column(elem_classes="question-group") as sample_question_component_group:
            empty_space_component = gr.HTML("", elem_classes="empty-div")
            default_dataset_label = gr.Markdown(
                "Default dataset is a sampling of articles recently published on GeForce News",
                elem_classes="description-secondary-markdown chat-disclaimer-message margin-"
            )
            with gr.Column(visible=False) as english_sample_questions:
                for i in range(0, len(sample_questions), 2):
                    row_questions = sample_questions[:2]
                    sample_questions = sample_questions[2:]
                    with gr.Row() as question_row:
                        for index, question in enumerate(row_questions):
                            query = question["query"]
                            button = gr.Button(
                                query,
                                elem_classes="sample-question-button"
                            )
                            question_butons.append({
                                "question": query,
                                "component": button
                            })
                            question_rows.append(question_row)
            with gr.Column(visible=False) as chinese_sample_questions:
                for i in range(0, len(sample_questions_chinese), 2):
                    row_questions = sample_questions_chinese[:2]
                    sample_questions_chinese = sample_questions_chinese[2:]
                    with gr.Row() as question_row:
                        for index, question in enumerate(row_questions):
                            query = question["query"]
                            button = gr.Button(
                                query,
                                elem_classes="sample-question-button"
                            )
                            question_butons.append({
                                "question": query,
                                "component": button
                            })
                            question_rows.append(question_row)
            with gr.Column(visible=False) as clip_sample_questions:
                for i in range(0, len(sample_questions_clip), 2):
                    row_questions = sample_questions_clip[:2]
                    sample_questions_clip = sample_questions_clip[2:]
                    with gr.Row() as question_row:
                        for index, question in enumerate(row_questions):
                            query = question["query"]
                            button = gr.Button(
                                query,
                                elem_classes="sample-question-button"
                            )
                            question_butons.append({
                                "question": query,
                                "component": button
                            })
                            question_rows.append(question_row)
        return question_butons, question_rows, empty_space_component, default_dataset_label, sample_question_component_group, english_sample_questions, chinese_sample_questions, clip_sample_questions

    def _render_chatbot(self, show_chatbot):
        chatbot_window = gr.Chatbot(
            show_label=False,
            elem_classes="chat-window",
            visible=show_chatbot,
            elem_id="main-chatbot-window",
            sanitize_html=True
        )
        isChatWithMicEnabled = self.config.get_config('models/enable_asr')
        with gr.Group() as query_group:
            with gr.Row():
                with gr.Group(elem_id="chat_box_group"):
                    with gr.Row():
                        query_input = gr.Textbox(placeholder="ChatRTX: Type or use voice", container=False, elem_id="chat_text_area")
                        chat_mic_component = gr.Audio(label="Microphone", sources=["upload", "microphone"], type="filepath", elem_id='microphone', render=isChatWithMicEnabled, visible=False)
                        gr.Button(
                            "",
                            scale=0,
                            icon=os.path.join(os.path.dirname(__file__), 'assets/mic_off.png'),
                            elem_classes="icon-button tooltip-component",
                            elem_id="mic-init-button",
                            render=isChatWithMicEnabled
                        )
                        mic_start_button = gr.Button(
                            "",
                            scale=0,
                            icon=os.path.join(os.path.dirname(__file__), 'assets/mic_off.png'),
                            elem_classes="icon-button tooltip-component",
                            elem_id="mic-off-button",
                            render=isChatWithMicEnabled
                        )
                        mic_stop_button = gr.Button(
                            "",
                            scale=0,
                            icon=os.path.join(os.path.dirname(__file__), 'assets/mic_on.png'),
                            elem_classes="icon-button tooltip-component",
                            elem_id="mic-on-button",
                            render=isChatWithMicEnabled
                        )
                        gr.Button(
                            "",
                            scale=0,
                            icon=os.path.join(os.path.dirname(__file__), 'assets/mic_blocked.png'),
                            elem_classes="icon-button tooltip-component",
                            elem_id="mic-blocked-button",
                            render=isChatWithMicEnabled
                        )
                submit_button = gr.Button("SEND", variant="primary", scale=0, elem_id="submit_button")
        with gr.Row() as chat_action_buttons_row:
            gr.HTML("")
            retry_button = gr.Button(
                "",
                elem_classes="icon-button tooltip-component",
                elem_id="chatbot-retry-button",
                scale=0,
                icon=os.path.join(os.path.dirname(__file__), 'assets/retry.png'),
            )
            undo_button = None
            if self.config.get_config_from_file("is_chat_engine", os.path.join(os.path.curdir, "config/app_config.json")) == False:
                undo_button = gr.Button(
                    "",
                    scale=0,
                    icon=os.path.join(os.path.dirname(__file__), 'assets/undo.png'),
                    elem_classes="icon-button tooltip-component",
                    elem_id="chatbot-undo-button"
                )
            reset_button = gr.Button(
                "",
                elem_classes="icon-button tooltip-component",
                elem_id="chatbot-reset-button",
                scale=0,
                icon=os.path.join(os.path.dirname(__file__), 'assets/reset.png'),
            )
            gr.HTML("")
        chat_disclaimer_markdown = gr.Markdown(
            "ChatRTX response quality depends on the AI model's accuracy and the input dataset. Please verify important information.",
            elem_classes="description-secondary-markdown chat-disclaimer-message margin-"
        )
        return (chatbot_window, query_input, chat_mic_component, mic_start_button, mic_stop_button, submit_button, retry_button, undo_button, reset_button, query_group, chat_disclaimer_markdown, chat_action_buttons_row)

    def _handle_events(self):
        self._handle_load_events()
        self._handle_shutdown_events()
        self._handle_settings_event()
        self._handle_model_events()
        self._handle_dataset_events()
        self._handle_chatbot_events()
        self._handle_mic_events()
        return None

    def _validate_session_and_raise(self, request: gr.Request):
        try:
            self._validate_request(request)
        except Exception as e:
            raise gr.Error('Invalid session')

    def _validate_session(self, request: gr.Request):
        try:
            self._validate_request(request)
        except Exception as e:
            return [
                gr.Group(visible=False),
                gr.Group(visible=False),
                gr.Group(visible=False),
                gr.Chatbot(visible=False),
                gr.Group(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Group(visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=True),
                gr.Markdown(visible=False),
                gr.Button(visible=False)
             ] + self._get_sample_question_components_new(True)
        return [
            gr.Group(),
            gr.Group(),
            gr.Group(),
            gr.Chatbot(),
            gr.Group(),
            gr.Button(),
            gr.Button(),
            gr.Group(),
            gr.Button(),
            gr.Button(),
            gr.Markdown(),
            gr.Markdown(),
            gr.Markdown(),
            gr.Button()
         ] + self._get_sample_question_components_new()
    
    def _get_validate_session_output(self):
        return [
            self._settings_page,
            self._model_group,
            self._dataset_group,
            self._chat_bot_window,
            self._chat_query_group,
            self._chat_reset_button,
            self._chat_retry_button,
            self._shutdown_post_shutdown_group,
            self._shutdown_button,
            self._chat_undo_button,
            self._chat_disclaimer_markdown,
            self._shutdown_invalid_session_markdown,
            self._shutdown_memory_released_markdown,
            self._settings_button
        ] + self._get_sample_question_components()

    def _handle_load_events(self):
        self._interface.load(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs()
        )
        return None

    def _handle_shutdown_events(self):
        def close_thread(session_id):
            if self._shutdown_callback:
                self._shutdown_callback(session_id)
            self._interface.close()
            self._interface = None
            print('exiting')
            os._exit(0)
            
        def handle_shutdown(state, request: gr.Request):
            self._validate_session(request)
            if self._interface is not None:
                _close_thread = threading.Thread(target=close_thread, args=(self._get_session_id(state),))
                _close_thread.start()
            else:
                print("Interface not initialized or already closed")
            return state
        def before_shutdown(request: gr.Request):
            self._validate_session(request)
            ret_val = [
                gr.Group(visible=False),
                gr.Group(visible=False),
                gr.Group(visible=False),
                gr.Chatbot(visible=False),
                gr.Group(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Group(visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=True),
                gr.Button(visible=False)
            ] + self._get_sample_question_components_new(True)
            return ret_val
        

        self._shutdown_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            before_shutdown,
            None,
            [
                self._settings_page,
                self._model_group,
                self._dataset_group,
                self._chat_bot_window,
                self._chat_query_group,
                self._chat_reset_button,
                self._chat_retry_button,
                self._shutdown_post_shutdown_group,
                self._shutdown_button,
                self._chat_undo_button,
                self._chat_disclaimer_markdown,
                self._shutdown_invalid_session_markdown,
                self._shutdown_memory_released_markdown,
                self._settings_button
            ] + self._get_sample_question_components()
        ).then(
            handle_shutdown,
            self._state,
            self._state
        )

    def _get_uninstall_components(self):
        components = []
        for key in self._uninstall_model_dict:
            components.append(self._uninstall_model_dict[key]['model_row']),
            components.append(self._uninstall_model_dict[key]['delete_button']),
            components.append(self._uninstall_model_dict[key]['model_name_html'])
            components.append(self._uninstall_model_dict[key]['model_name_with_active_tag_html'])
        return components
    
    def _set_uninstall_components(self):
        ret_val = []
        for key in self._uninstall_model_dict:
            isVisible = key in self._installed_models
            isAcitve = key == self._installed_dropdown_selected_model
            ret_val.append(gr.Row(visible=isVisible))
            ret_val.append(gr.Button(interactive=not isAcitve))
            ret_val.append(gr.HTML(visible=not isAcitve))
            ret_val.append(gr.HTML(visible=isAcitve))
        return ret_val

    def _handle_settings_event(self):

        def check_for_setting_enablement(model_name, request: gr.Request):
            self._validate_session(request)
            if str(model_name).startswith('Downloading'):
                raise gr.Warning('AI model download in progress - Retry once done')

        def on_settings_clicked(request: gr.Request):
            self._validate_session(request)
            return self._set_uninstall_components() + [
                gr.Column(visible=False),
                gr.Column(visible=True),
                gr.Dropdown(choices=[],value=None),
                gr.Dropdown(choices=[],value=None)
            ]
        
        def check_for_settings_closed_enablement(request: gr.Request):
            self._validate_session(request)
            if self._model_uninstall_progress_counter != 0:
                raise gr.Warning("AI model is uninstalling - Retry once done")

        def on_settings_close_clicked(request: gr.Request):
            self._validate_session(request)
            self.update_model_info()
            return [
                gr.Column(visible=True),
                gr.Column(visible=False),
                gr.Accordion(visible= (self._ready_to_install_model is not None or self._download_dropdown_selected_model is not None)),
                gr.Group(visible = (self._ready_to_install_model is not None)),
                gr.Group(visible= (self._ready_to_install_model is None and self._download_dropdown_selected_model is not None)),
                gr.Dropdown(choices=[],value=None) if self._ready_to_install_model is not None else \
                    gr.Dropdown(self.get_formatted_model_name_with_size(self._available_to_download_models), value=self.get_formatted_model_name_with_size(self._download_dropdown_selected_model), visible=True),
                gr.Dropdown(self._installed_models, value = self._installed_dropdown_selected_model, visible=True),
                gr.Markdown("" if not self._ready_to_install_model else self._models_info_map[self._ready_to_install_model]["model_info"]) if self._ready_to_install_model is not None \
                    else gr.Markdown("" if not self._download_dropdown_selected_model else self._models_info_map[self._download_dropdown_selected_model]["model_info"]),
                gr.Markdown("" if not self._ready_to_install_model else self._models_info_map[self._ready_to_install_model]["model_license"]) if self._ready_to_install_model is not None  \
                    else gr.Markdown("" if not self._download_dropdown_selected_model else self._models_info_map[self._download_dropdown_selected_model]["model_license"]),
                gr.Row(visible= (self._ready_to_install_model is None and self._download_dropdown_selected_model is not None))
            ]
        
        def on_delete_model_click(model_name, request: gr.Request):
            self._validate_session(request)
            if not model_name:
                return gr.Button(), gr.Button(), gr.Row()
            self._model_uninstall_progress_counter += 1
            isSuccess = False
            if self._on_model_delete_callback:
                isSuccess = self._on_model_delete_callback(self._models_info_map[model_name])
            if isSuccess:
                gr.Info(model_name + " deleted")
                model_list = self.config.read_default_config(self._models_path_key + '/supported')
                for index, model in enumerate(model_list):
                    if model_name == model["name"]:
                        model_list[index]["setup_finished"] = False
                        model_list[index]["downloaded"] = False
                self.config.write_default_config(self._models_path_key + '/supported', model_list)
                self._model_uninstall_progress_counter -= 1
            else:
                self._model_uninstall_progress_counter -= 1
                raise gr.Error(model_name + " uninstall failed!")
            return gr.Button(visible=False), gr.Button(visible=True), gr.Row(visible=False)
    
    
        def before_model_delete(request: gr.Request):
            self._validate_session(request)
            return gr.Button(visible=True), gr.Button(visible=False)

        confirm_delete_js = """(value) => {
            const [check_box, model_name] = arguments[0];
            if (confirm('Are You sure you want to delete ' + model_name + '?') == true) {
                return !value;
            } else {
                return value;
            }
        }"""

        self._settings_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            check_for_setting_enablement,
            self._model_name_markdown,
            None
        ).success(
            on_settings_clicked,
            None,
            self._get_uninstall_components()+ [
                self._chat_page,
                self._settings_page,
                self._download_model_dropdown,
                self._model_dropdown,
            ]
        )

        self._settings_close_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            check_for_settings_closed_enablement,
            None,
            None
        ).success(
            on_settings_close_clicked,
            None,
            [
                self._chat_page,
                self._settings_page,
                self._add_model_accordion,
                self._install_model_group,
                self._download_model_group,
                self._download_model_dropdown,
                self._model_dropdown,
                self._model_info_markdown,
                self._model_license_markdown,
                self._model_license_row
            ]
        )

        for key in self._uninstall_model_dict:
            _hidden_checkbox = gr.Checkbox(visible=False, interactive=True)
            self._uninstall_model_dict[key]['delete_button'].click(
                self._validate_session,
                None,
                self._get_validate_session_output()
            ).then(
                self._validate_session_and_raise,
                None,
                None
            ).success(
                None,
                [_hidden_checkbox, self._uninstall_model_dict[key]['model_name_html']],
                _hidden_checkbox,
                js=confirm_delete_js
            )

            _hidden_checkbox.change(
                self._validate_session,
                None,
                self._get_validate_session_output()
            ).then(
                self._validate_session_and_raise,
                None,
                None
            ).success(
                before_model_delete,
                None,
                [self._uninstall_model_dict[key]['progress_spinner_button'], self._uninstall_model_dict[key]['delete_button']],
            ).then(
                on_delete_model_click,
                self._uninstall_model_dict[key]['model_name_html'],
                [self._uninstall_model_dict[key]['progress_spinner_button'], self._uninstall_model_dict[key]['delete_button'], self._uninstall_model_dict[key]['model_row']]
            )

    def get_formatted_model_name_with_size(self, val):
        if isinstance(val, list):
            res = []
            for v in val:
                res.append(self.get_formatted_model_name_with_size(v))
            return res
        elif val is not None:
            if val != 'None' and self._models_info_map[val] is not None:
                return val + ' (' + self._models_info_map[val]["model_size"] + ')'
        return val

    def get_model_name_from_formatted_name_with_size(self, val):
        if type(val) == list():
            res = []
            for v in val:
                res.append(self.get_model_name_from_formatted_name_with_size(v))
            return res
        elif val is not None:
            pattern = r' \(\d+(\.\d+)?[KMGT]B\)'
            ret_val = re.sub(pattern, '', val)
            if val != 'None' and self._models_info_map[ret_val] is not None:
                return ret_val
        return val

    def update_model_info(self):
            models_data = self.config.read_default_config(self._models_path_key)
            models = sorted(models_data['supported'], key=lambda d: d['name'])
            models = [model for model in models if model['should_show_in_UI']]
            self._installed_models = []
            self._ready_to_install_model = None
            self._available_to_download_models = []
            for model in models:
                if model['downloaded'] is True and model['setup_finished'] is True:
                    self._installed_models.append(model['name'])
                elif model['downloaded'] is True and model['setup_finished'] is False and not self._ready_to_install_model:
                    self._ready_to_install_model = model['name']
                elif model['downloaded'] is False:
                    self._available_to_download_models.append(model['name'])
                self._models_info_map[model['name']] = model

            self._installed_dropdown_selected_model = self.config.get_config('models/selected')
            if len(self._installed_models) > 0:
                if self._installed_dropdown_selected_model not in self._installed_models:
                    self._installed_dropdown_selected_model = self._installed_models[0]
                    self.config.set_config('models/selected', self._installed_dropdown_selected_model)
            else:
                self._installed_dropdown_selected_model = None

            self._download_dropdown_selected_model = self.config.get_config('models/selected_for_download')
            if len(self._available_to_download_models) > 0:
                if self._download_dropdown_selected_model not in self._available_to_download_models:
                    self._download_dropdown_selected_model = self._available_to_download_models[0]
                    self.config.set_config('models/selected_for_download', self._download_dropdown_selected_model)
            else:
                self._download_dropdown_selected_model = None

    def _handle_model_events(self):
        def change_installed_model(newModel, state):
            if self._model_change_callback:
                self._model_change_callback(
                    self._models_info_map[newModel]['name'],
                    self._models_info_map[newModel],
                    self._get_session_id(state)
                )
            self.config.set_config("models/selected", newModel)

        def on_selection_change(newModel, state, request: gr.Request):
            self._validate_session(request)
            if not newModel or str(newModel) == 'None' or newModel is None or str(newModel) == self._installed_dropdown_selected_model:
                return newModel, state
            change_installed_model(newModel, state)
            self._installed_dropdown_selected_model = str(newModel)
            return newModel, state
        
        def update_dataset_picker_helper(newModel):
            sources = copy.deepcopy(self.config.get_config("dataset/sources"))
            self._dataset_selected_source = self.config.get_config("dataset/selected")
            markdown_text = self._dataset_directory_label_text
            if newModel == self._clip_model_name:
                sources.remove("nodataset")
                markdown_text = self._data_directory_clip_label_text
                if (self._dataset_selected_source == "nodataset"):
                    self._dataset_selected_source = "directory"
            return gr.Markdown(markdown_text), gr.Dropdown(self.config.get_display_strings(sources),
                    value=self.config.get_display_strings(self._dataset_selected_source))

        def update_dataset_picker(newModel, request: gr.Request):
            self._validate_session(request)
            if not newModel:
                return gr.Markdown(), gr.Dropdown()
            return update_dataset_picker_helper(newModel)

        def on_download_selection_change(newModel, request:gr.Request):
            self._validate_request(request)
            model_name = self.get_model_name_from_formatted_name_with_size(str(newModel))
            if model_name == 'None' or model_name is None:
                return '', '', gr.Checkbox(value=False)
            if (model_name):
                self.config.set_config("models/selected_for_download", model_name)
                self.update_model_info()
            info = '' if not model_name else self._models_info_map[str(model_name)]["model_info"]
            license = '' if not model_name else self._models_info_map[str(model_name)]["model_license"]
            return info, license, gr.Checkbox(value=False)
        
        def check_agreement(isAgreed):
            if not isAgreed:
                raise gr.Warning("Read and accept the applicable terms, before Downloading the AI model")
            return
    
        def start_model_download():
            return [
                gr.Group(visible=False),
                gr.Group(visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Markdown("Downloading " + str(self._download_dropdown_selected_model) + "..."),
                gr.Row(visible=False)
            ]

        def on_model_download_click(newModel, request: gr.Request):
            self._validate_request(request)
            is_download_success = False
            model_name = self.get_model_name_from_formatted_name_with_size(str(newModel))
            if self._on_model_downloaded_callback:
                is_download_success = self._on_model_downloaded_callback(self._models_info_map[model_name])
            if is_download_success:
                model_list = self.config.read_default_config(self._models_path_key + '/supported')
                for index, model in enumerate(model_list):
                    if model_name == model["name"]:
                        model_list[index]["downloaded"] = True
                self.config.write_default_config(self._models_path_key + '/supported', model_list)
                self.update_model_info()
                print("Model ready to install ", self._ready_to_install_model)
                info_msg = self._ready_to_install_model + " download complete - Ready to install"
                gr.Info(info_msg)
            else:
                raise gr.Error( model_name + " failed to download")


        def check_if_model_download(request: gr.Request):
            self._validate_session(request)
            if self._ready_to_install_model is not None:
                markdown_str = str(self._ready_to_install_model)
                return [
                    gr.Group(visible=True),
                    gr.Button(visible=True),
                    gr.Button(visible=True),
                    gr.Markdown(markdown_str),
                    gr.Group(visible=False),
                    gr.Dropdown(),
                    gr.Row(visible=False)
                ]
            return [
                gr.Group(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Markdown(""),
                gr.Group(visible=True),
                gr.Dropdown(self.get_formatted_model_name_with_size(self._available_to_download_models), value=self.get_formatted_model_name_with_size(self._download_dropdown_selected_model)),
                gr.Row(visible=True)
            ]

        def before_install(request: gr.Request):
            self._validate_session(request)
            ret_val = [
                gr.Group(visible=False),
                gr.Group(visible=False),
                gr.Group(visible=True),
                gr.Markdown(self._ready_to_install_model + " Installing"),
                gr.Chatbot(visible=False),
                gr.Group(visible=False),
                gr.Row(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Markdown(visible=False),
                gr.Accordion(open=False)
            ] + self._get_sample_question_components_new(True)
            return ret_val

        def on_model_install_click(state, request: gr.Request):
            self._validate_session(request)
            is_success = False
            if self._on_model_installed_callback:
                is_success = self._on_model_installed_callback(self._models_info_map[self._ready_to_install_model])
            if (is_success):
                self.config.set_config("models/selected", self._ready_to_install_model)
                model_list = self.config.read_default_config(self._models_path_key + '/supported')
                for index, model in enumerate(model_list):
                    if self._ready_to_install_model == model["name"]:
                        model_list[index]["setup_finished"] = True
                self.config.write_default_config(self._models_path_key + '/supported', model_list)
                print("Model installed ", self._ready_to_install_model)
                change_installed_model(self._ready_to_install_model, state)
                self.update_model_info()
                return update_dataset_picker_helper(self._installed_dropdown_selected_model)
            else:
                raise gr.Error(f'{ self._ready_to_install_model } failed to install')


        def post_install(state, request: gr.Request):
            self._validate_session(request)
            if self._reset_chat_callback:
                self._reset_chat_callback(self._get_session_id(state))
            ret_val = [
                state,
                gr.Group(visible=True),
                gr.Group(visible=True),
                gr.Group(visible=False),
                gr.Button(visible=True),
                gr.Button(visible=True),
                gr.Chatbot([],visible=True),
                gr.Group(visible=True),
                gr.Row(visible=True),
                gr.Markdown(visible=True),
                gr.Accordion(visible= (self._ready_to_install_model is not None or self._download_dropdown_selected_model is not None)),
                gr.Group(visible = (self._ready_to_install_model is not None)),
                gr.Group(visible= (self._ready_to_install_model is None and self._download_dropdown_selected_model is not None)),
                gr.Dropdown(self.get_formatted_model_name_with_size(self._available_to_download_models), value=self.get_formatted_model_name_with_size(self._download_dropdown_selected_model), visible=True),
                gr.Dropdown(self._installed_models, value = self._installed_dropdown_selected_model, visible=True),
                gr.Markdown("" if not self._download_dropdown_selected_model else self._models_info_map[self._download_dropdown_selected_model]["model_info"]),
                gr.Markdown("" if not self._download_dropdown_selected_model else self._models_info_map[self._download_dropdown_selected_model]["model_license"]),
                gr.Row(visible= (self._ready_to_install_model is None and self._download_dropdown_selected_model is not None))
            ] + self._get_sample_question_components_new(True)
            return ret_val
        
        def cancel_install(request: gr.Request):
            self._validate_session(request)
            model_list = self.config.read_default_config(self._models_path_key + '/supported')
            for index, model in enumerate(model_list):
                if self._ready_to_install_model == model["name"]:
                    model_list[index]["downloaded"] = False
            self.config.write_default_config(self._models_path_key + '/supported', model_list)
            self.config.set_config("models/selected_for_download", self._ready_to_install_model)
            self.update_model_info()
            return [
                gr.Group(visible = False),
                gr.Group(visible= True),
                gr.Dropdown(self.get_formatted_model_name_with_size(self._available_to_download_models), value=self.get_formatted_model_name_with_size(self._download_dropdown_selected_model), visible=True),
                gr.Markdown(self._models_info_map[self._download_dropdown_selected_model]["model_info"]),
                gr.Markdown(self._models_info_map[self._download_dropdown_selected_model]["model_license"]),
                gr.Row(visible= True)
            ]
    
        _js_post_install_accordion_wrap = """
            () => {
                setTimeout(() => {
                    const accordionWrapButton = document.querySelector(".add-model-accordion > button");
                    accordionWrapButton?.click();
                }, 100)
            }
        """


        self._model_dropdown.input(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_change_element_state,
            None,
            self._get_enable_disable_elemet_list() + self._get_sample_question_components()
        ).then(
            on_selection_change,
            [self._model_dropdown, self._state],
            [self._model_dropdown, self._state]
        ).then(
            update_dataset_picker,
            self._model_dropdown,
            [self._dataset_label_markdown, self._dataset_source_dropdown]
        ).then(
            self._after_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        ).then(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs()
        )

        self._download_model_dropdown.change(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            on_download_selection_change,
            self._download_model_dropdown,
            [self._model_info_markdown, self._model_license_markdown, self._license_agreement]
        )

        self._download_model_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            check_agreement,
            self._license_agreement,
            None
        ).success(
            start_model_download,
            None,
            [self._download_model_group ,self._install_model_group, self._cancel_install_btn, self._install_model_btn, self._model_name_markdown, self._model_license_row]
        ).then(
            on_model_download_click,
            self._download_model_dropdown,
            None
        ).then(
            check_if_model_download,
            None,
            [self._install_model_group, self._install_model_btn, self._cancel_install_btn, self._model_name_markdown, self._download_model_group, self._download_model_dropdown, self._model_license_row]
        )


        _cofirm_js = """(x) => {
            if (confirm('Proceeding with the installation will change the model, reset the chat, and activate the new model.') == true) {
                return !x;
            } else {
                return x
            }
        }"""


        self._install_model_btn.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            None,
            self._hidden_checkbox,
            self._hidden_checkbox,
            js=_cofirm_js
        )


        self._hidden_checkbox.change(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            before_install,
            None,
            [
                self._model_group,
                self._dataset_group,
                self._install_progress_group,
                self._install_progress_markdown,
                self._chat_bot_window,
                self._chat_query_group,
                self._chat_action_buttons_row,
                self._shutdown_button,
                self._settings_button,
                self._chat_disclaimer_markdown,
                self._add_model_accordion
            ] + self._get_sample_question_components()
        ).then(
            on_model_install_click,
            self._state,
            [self._dataset_label_markdown, self._dataset_source_dropdown]
        ).then(
            post_install,
            self._state,
            [
                self._state,
                self._model_group,
                self._dataset_group,
                self._install_progress_group,
                self._shutdown_button,
                self._settings_button,
                self._chat_bot_window,
                self._chat_query_group,
                self._chat_action_buttons_row,
                self._chat_disclaimer_markdown,
                self._add_model_accordion,
                self._install_model_group,
                self._download_model_group,
                self._download_model_dropdown,
                self._model_dropdown,
                self._model_info_markdown,
                self._model_license_markdown,
                self._model_license_row
            ] + self._get_sample_question_components(),
        ).then(
            None,
            None,
            None,
            js=_js_post_install_accordion_wrap
        )

        self._cancel_install_btn.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            cancel_install,
            None,
            [
                self._install_model_group,
                self._download_model_group,
                self._download_model_dropdown,
                self._model_info_markdown,
                self._model_license_markdown,
                self._model_license_row
            ]
        )

    
    def _handle_dataset_events(self):
        def select_folder(path, state, request: gr.Request):
            self._validate_session(request)
            if self._dataset_selected_source == "directory":
                command = [sys.executable, "./ui/select_folder.py"]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, _ = process.communicate()
                # Check if the command was successful
                result_string = ""
                if process.returncode == 0:
                    result_string = output.decode().strip()
                else:
                    print("Error executing script:", process.returncode)
                if len(result_string) > 0:
                    self._dataset_path = result_string
                    self.config.set_config(self._dataset_path_key, {"path": self._dataset_path, "isRelative": False})
            else:
                self._dataset_path = path

            if self._dataset_path_updated_callback:
                self._dataset_path_updated_callback(
                    self._dataset_selected_source,
                    self._dataset_path,
                    None,
                    self._get_session_id(state)
                )
            return self._dataset_path, state
        
        self._dataset_update_source_edit_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_change_element_state,
            None,
            self._get_enable_disable_elemet_list() + self._get_sample_question_components()
        ).then(
            select_folder,
            [self._dataset_source_textbox, self._state],
            [self._dataset_source_textbox, self._state]
        ).then(
            self._after_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        ).then(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs(),
            show_progress=False
        )

        def on_dataset_source_changed(source, state, request: gr.Request):
            self._validate_session(request)
            self._dataset_selected_source = self.config.get_display_string_keys(source)
            source = self._dataset_selected_source
            self.config.set_config("dataset/selected", source)
            self._dataset_path = self._get_dataset_path() if source=="directory" else ""
            if self._dataset_source_updated_callback:
                self._dataset_source_updated_callback(
                    self._dataset_selected_source,
                    self._dataset_path,
                    self._get_session_id(state)
                )
            return [
                gr.Textbox(
                    interactive=False,
                    visible=source!="nodataset",
                    value=self._dataset_path
                ),
                gr.Button(visible=source=="directory"),
                gr.Button(visible=source!="nodataset"),
                state
            ]

        self._dataset_source_dropdown.change(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        ).then(            
            on_dataset_source_changed,
            [self._dataset_source_dropdown, self._state],
            [
                self._dataset_source_textbox,
                self._dataset_update_source_edit_button,
                self._dataset_regenerate_index_button,
                self._state
            ],
            show_progress=False
        ).then(
            self._after_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        ).then(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs(),
            show_progress=False
        )

        def regenerate_index(state, request: gr.Request):
            self._validate_session(request)
            if self._regenerate_index_callback:
                self._regenerate_index_callback(self._dataset_selected_source, self._dataset_path, self._get_session_id(state))
            return self._dataset_path, state

        self._dataset_regenerate_index_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_change_element_state,
            None,
            self._get_enable_disable_elemet_list()
        ).then(
            regenerate_index,
            self._state,
            [self._dataset_source_textbox, self._state]
        ).then(
            self._after_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        )


# dataset events ends
    def _show_hide_sample_questions(self, query, history, dataset_source, model_dropdown, state, request: gr.Request):
        self._validate_session(request)
        dataset_source = self.config.get_display_string_keys(dataset_source)
        # sample_question_shown = state['sample_question_shown'] if isinstance(state, object) and 'sample_question_shown' in state else False
        hide_sample_ques = \
            len(query) > 0 or \
            len(history) > 0 or \
            ((os.path.normpath(self._dataset_path) != os.path.normpath(self._default_dataset_path)) and \
             (os.path.normpath(self._dataset_path) != os.path.normpath(self._default_clip_dataset_path)) and \
             os.path.normpath(self._dataset_path) != os.path.normpath(self._default_chinese_dataset_path)) or \
             (os.path.normpath(self._dataset_path) == os.path.normpath(self._default_chinese_dataset_path) and model_dropdown != self._chinese_model_name) or \
             (os.path.normpath(self._dataset_path) == os.path.normpath(self._default_dataset_path) and (model_dropdown == self._chinese_model_name or model_dropdown == self._clip_model_name)) or \
             (os.path.normpath(self._dataset_path) == os.path.normpath(self._default_clip_dataset_path) and model_dropdown != self._clip_model_name)
        # if isinstance(state, object):
        #     state['sample_question_shown'] = hide_sample_ques
        ret_val = [gr.Chatbot(history, visible=hide_sample_ques), gr.Column(visible=not hide_sample_ques)]
        ret_val.append(gr.Column(visible= ((model_dropdown != self._chinese_model_name or model_dropdown != self._clip_model_name) and os.path.normpath(self._dataset_path) == os.path.normpath(self._default_dataset_path))))
        ret_val.append(gr.Column(visible=(model_dropdown == self._chinese_model_name and os.path.normpath(self._dataset_path) == os.path.normpath(self._default_chinese_dataset_path))))
        ret_val.append(gr.Column(visible=(model_dropdown == self._clip_model_name and os.path.normpath(self._dataset_path) == os.path.normpath(self._default_clip_dataset_path))))
        ret_val.append(state)
        return ret_val
    
    def _get_show_hide_sample_questions_inputs(self):
        return [
            self._chat_query_input_textbox ,self._chat_bot_window, self._dataset_source_dropdown, self._model_dropdown ,self._state
        ]
    
    def _get_sample_question_components_new(self, hide_sample_ques: bool = None):
        if hide_sample_ques is None: # neither show nor hide
            ret_val = [gr.Column()]
        else:
            ret_val = [gr.Column(visible=not hide_sample_ques)]
        return ret_val
    
    def _get_sample_question_components(self):
        return [self._sample_question_component_group]

    def _get_show_hide_sample_questions_outputs(self):
        return [self._chat_bot_window, self._sample_question_component_group, self._english_sample_questions, self._chinese_sample_questions, self._clip_sample_questions, self._state]

# chat bot events
    def _handle_chatbot_events(self):
        def process_input(query, history, request: gr.Request):
            self._validate_session(request)
            if len(query) == 0:
                return "", history
            history.append([query, None])
            return "", history
        
        def process_output(history, state, request: gr.Request):
            self._validate_session(request)
            if len(history) == 0:
                yield history, state
            else:
                query = history[-1]
                if query[1] != None:
                    yield history, state
                elif self._query_handler:
                    for response in self._query_handler(query[0], history[:-1], self._get_session_id(state)):
                        history[-1][1] = response
                        yield history, state
                else:
                    history[-1][1] = "ChatBot not ready..."
                    yield history, state
            
        #undo handler
        def process_undo_last_chat(history: list, state, request: gr.Request):
            self._validate_session(request)
            if len(history) == 0:
                return history, state

            history = history[:len(history) - 1]
            if self._undo_last_chat_callback:
                self._undo_last_chat_callback(history, self._get_session_id(state))
            
            return history, state

        #retry handler
        def process_retry(history: list, request: gr.Request):
            self._validate_session(request)
            if len(history) == 0:
                return history

            lastChat = history[-1]
            history = history[:len(history) - 1]
            _, history = process_input(lastChat[0], history, request)
            return history

        def reset(state, request: gr.Request):
            self._validate_session(request)
            if self._reset_chat_callback:
                self._reset_chat_callback(self._get_session_id(state))
            return "", [], state

        gr.on(
            [self._chat_query_input_textbox.submit, self._chat_submit_button.click],
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_state_when_response_processing,
            None,
            self._element_list_when_response_under_process()
        ).success(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs()
        ).then(
            process_input,
            [self._chat_query_input_textbox, self._chat_bot_window], 
            [self._chat_query_input_textbox, self._chat_bot_window]
        ).then(
            process_output,
            [self._chat_bot_window, self._state],
            [self._chat_bot_window, self._state]
        ).then(
            self._after_state_when_response_processing,
            None,
            self._element_list_when_response_under_process(),
            show_progress=False
        )

        self._chat_retry_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            process_retry,
            [self._chat_bot_window],
            [self._chat_bot_window]
        ).then(
            process_output,
            [self._chat_bot_window, self._state],
            [self._chat_bot_window, self._state]
        )

        if self._chat_undo_button:
            self._chat_undo_button.click(
                self._validate_session,
                None,
                self._get_validate_session_output()
            ).then(
                self._validate_session_and_raise,
                None,
                None
            ).success(
                process_undo_last_chat,
                [self._chat_bot_window, self._state],
                [self._chat_bot_window, self._state]
            )
        self._chat_reset_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            reset, 
            self._state,
            [self._chat_query_input_textbox, self._chat_bot_window, self._state]
        )

        def handle_sample_question_click(evt: gr.EventData, request: gr.Request):
            self._validate_session(request)
            return evt.target.value
        
        for sample in self._sample_question_components:
            button: gr.Button = sample['component']
            button.click(
                self._validate_session,
                None,
                self._get_validate_session_output()
            ).then(
                self._validate_session_and_raise,
                None,
                None
            ).success(
                self._before_state_when_response_processing,
                None,
                self._element_list_when_response_under_process()
            ).then(
                handle_sample_question_click,
                None,
                self._chat_query_input_textbox
            ).then(
                self._show_hide_sample_questions,
                self._get_show_hide_sample_questions_inputs(),
                self._get_show_hide_sample_questions_outputs()
            ).then(
                process_input,
                [self._chat_query_input_textbox, self._chat_bot_window], 
                [self._chat_query_input_textbox, self._chat_bot_window]
            ).then(
                process_output,
                [self._chat_bot_window, self._state],
                [self._chat_bot_window, self._state]
            ).then(
                self._after_state_when_response_processing,
                None,
                self._element_list_when_response_under_process(),
                show_progress=False
            )

    #Handle mic events
    def _handle_mic_events(self):

        def on_mic_button_click(request: gr.Request):
            self._validate_session(request)
            self._is_asr_model_loaded = False
            if (self._on_mic_button_click_callback):
                self._is_asr_model_loaded = self._on_mic_button_click_callback()
            if (self._is_asr_model_loaded):
                return gr.HTML("on")
            return gr.HTML("off")

        def on_recording_stop(audio_path, request: gr.Request):
            self._validate_session(request)
            value = ""
            if (self._on_mic_recording_done_callback):
                value = self._on_mic_recording_done_callback(audio_path)
            return gr.Textbox(value=value, interactive=True)
        
        def asr_model_check_and_stop_mic():
            return """
            () => {
                let asrState = document.getElementById("asr_model_state")?.innerText?.trim();
                console.log(asrState);
                if (asrState === "off") {
                    let micOnButton = document.getElementById("mic-on-button");
                    console.log("Recording stopped")
                    micOnButton.click();
                }
            }
        """

        def check_asr_model_load(request: gr.Request):
            self._validate_request(request)
            if (not self._is_asr_model_loaded):
                raise gr.Error("There was a problem recording audio")
            
        def disable_TextBox(request: gr.Request):
            self._validate_request(request)
            return gr.Textbox(value="", interactive=False)
        
        def enable_TextBox(request: gr.Request):
            self._validate_request(request)
            return gr.Textbox(interactive=True)
        
        self._mic_start_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_state_when_response_processing,
            None,
            self._element_list_when_response_under_process()
        ).then(
            disable_TextBox,
            None,
            self._chat_query_input_textbox
        ).then(
            on_mic_button_click,
            None,
            self._asr_model_state
        ).then(
            check_asr_model_load,
            None,
            None,
            js=asr_model_check_and_stop_mic()
        )

        self._mic_stop_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            enable_TextBox,
            None,
            self._chat_query_input_textbox
        )

        self._chat_mic_component.stop_recording(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).then(
            disable_TextBox,
            None,
            self._chat_query_input_textbox
        ).then(
            on_recording_stop,
            self._chat_mic_component,
            self._chat_query_input_textbox
        ).then(
            enable_TextBox,
            None,
            self._chat_query_input_textbox
        ).then(
            self._after_state_when_response_processing,
            None,
            self._element_list_when_response_under_process(),
            show_progress=False
        )
        
        return None