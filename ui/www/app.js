// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
() => {
    // file to customize behavior
    console.log('js loaded', window.location.href);

    //handle launch
    let reload = false;
    let gradioURL = new URL(window.location.href);
    if(gradioURL.searchParams.has('cookie')) {
        const secureCookie = gradioURL.searchParams.get('cookie');
        gradioURL.searchParams.delete('cookie');
        // to do set the cookie
        const SESSION_COOKIE_EXPIRY = 60 * 60 * 24; // 1 day
        //document.cookie = `_s_chat_=${secureCookie}; path=${location.pathname}; max-age=${SESSION_COOKIE_EXPIRY}; samesite=strict`;
        document.cookie = `_s_chat_=${secureCookie}; path=${location.pathname}; samesite=strict`;
        reload = true;
    }
    if(
        !gradioURL.searchParams.has('__theme') ||
        (gradioURL.searchParams.has('__theme') && gradioURL.searchParams.get('__theme') !== 'dark')
    ) {
        gradioURL.searchParams.delete('__theme');
        gradioURL.searchParams.set('__theme', 'dark');
        reload = true;
    }
    if(reload) {
        window.location.replace(gradioURL.href);
    }


    //tool tip
    const elements = document.getElementsByClassName("tooltip-component");
    console.log('elements found', elements.length)
    Array.prototype.forEach.call(elements, function (element) {
        // Get the tooltip content based on the element's ID
        let tooltipContent = null;

        if (element.id === "shutdown-btn") {
            tooltipContent ="Shutdown";
        } else if (element.id === "dataset-regenerate-index-btn") {
            tooltipContent ="Refresh dataset";
        } else if (element.id === "dataset-update-source-edit-button") {
            tooltipContent ="Change folder path";
        } else if (element.id === "chatbot-retry-button") {
            tooltipContent ="Regenerate response";
        } else if (element.id === "chatbot-reset-button") {
            tooltipContent ="Clear chat";
        } else if (element.id === "chatbot-undo-button") {
            tooltipContent ="Undo";
        } else if (element.id === "dataset-view-transcript-folder-button") {
            tooltipContent ="View index folder";
        } else if (element.id === "mic-off-button") {
            tooltipContent ="Click for voice input (30s max)";
        } else if (element.id === "mic-blocked-button") {
            tooltipContent ="Allow microphone access from browser";
        } else if (element.id === "mic-on-button") {
            tooltipContent ="Click to stop recording";
        } else if (element.id === "mic-init-button") {
            tooltipContent ="Allow microphone access"
        } else if (element.id === "download_model_button") {
            tooltipContent ="Download model"
        } else if (element.id === "settings-btn") {
            tooltipContent ="Settings"
        } else if (element.id === "settings-close-btn") {
            tooltipContent ="Close settings"
        } else if (element.id.startsWith("delete-model-btn")) {
            tooltipContent ="Delete AI model"
        } else if (element.id.startsWith("delete-model-progress-btn")) {
            tooltipContent ="Deleting AI model..."
        }

        console.log('tooltip content', tooltipContent);
        if(tooltipContent) {
            const tooltip = document.createElement("div");
            // var tooltipContentElement = document.getElementById("tooltip-content-element");
            // tooltipContentElement.forEach(className => {
            //     tooltip.classList.add(className);
            // });
            tooltip.classList.add("tooltip");
            tooltip.style.display = "none";
            tooltip.innerHTML = tooltipContent;
            document.body.appendChild(tooltip);
    
            element.addEventListener("mouseover", function () {
                console.log('mouse over', elements.length);
                var rect = element.getBoundingClientRect();
                tooltip.style.left = rect.left + "px";
                tooltip.style.top = (rect.bottom + 5) + "px";
                tooltip.style.display = "block";
            });
    
            element.addEventListener("mouseout", function () {
                tooltip.style.display = "none";
            });
        }
    });

    // links in chatbot window
    let el = document.getElementById("main-chatbot-window");
    if(el) {
        el.addEventListener("click", (event) => {
            if (event.target.getAttribute("data-link")) {
                data = {
                    data: [event.target.getAttribute("data-link")]
                }
                makeHttpCall('/api/open_local_link', 'POST', {}, data)
            }
        })
    } else {
        console.error("chatbot window not created");
    }

    function makeHttpCall(url, method, headers, data) {
        headers = Object.keys(headers).length !== 0 ? headers : {
            "Content-Type": "application/json"
        };
        options = {
            method,
            headers,
            body: JSON.stringify(data)
        };
        console.log(options);
        fetch(url, options)
            .then(response => response.json())
            .then(data => {
                console.log("Response:", data);
            })
            .catch(error => {
                console.error("Error:", error);
            });
    }

    let micInitButton = document.getElementById("mic-init-button");
    let micOffButton = document.getElementById("mic-off-button");
    let micOnButton = document.getElementById("mic-on-button");
    let micBlockedButton = document.getElementById("mic-blocked-button");
    let chatBoxGroup = document.getElementById("chat_box_group");
    let textBox = document.querySelector("#chat_text_area > label > textarea");
    let recordingState  = null;
    let micComponent = document.getElementById("microphone");

    navigator.permissions.query(
        { name: 'microphone' }
    ).then(function(permissionStatus){
        console.log("checking mic permission ", permissionStatus.state); // granted, denied, prompt
        // In case of prompt set to mic init state
        // Init mic button ask for mic permssions
        if (permissionStatus.state === "prompt") {
            setMicButtonState("init");
        } else if (permissionStatus.state === "granted") {
            setMicMode();
            setMicButtonState("off");
        } else if (permissionStatus.state === "denied") {
            setMicButtonState("blocked");
        }
    });

    // Mic Init button to ask for mic permissions
    micInitButton?.addEventListener("click", (_) => {
        micPermissionPrompt()
    })
    
    micOffButton?.addEventListener("click", (_) => {
        textBox.value = "";
        navigator.permissions.query(
            { name: 'microphone' }
        ).then(function(permissionStatus){
            console.log("checking mic permission ", permissionStatus.state); // granted, denied, prompt
            // Here state would either be granted or blocked, blocked in case user manually blocks mic
            if (permissionStatus.state === "granted") {
                // Start recoding
                toggleRecord();
            } else {
                // Show mic not found here
                console.log('Mic permissions denied');
                setMicButtonState("blocked");
            }
        })
    })

    
    micBlockedButton?.addEventListener("click", (_) => {
        navigator.permissions.query(
            { name: 'microphone' }
        ).then(function(permissionStatus){
            if (permissionStatus.state === "granted") {
                recordingState = "off";
                setMicMode();
                toggleRecord();
            }
        })
    })


    micOnButton?.addEventListener("click", () => {
        recordingState="on"
        toggleRecord();
    })


    function micPermissionPrompt() {
        navigator.getUserMedia({ audio: true }, function (e) {
            // Mic permission granted
            // Set mic state off
            // Set gradio audio component to mic
            setMicMode()
            setMicButtonState("off");
        }, function (err) {
            if (err.name === 'PermissionDismissedError') {
                console.log('Mic permissions dismissed');
            }
            if (err.name === 'PermissionDeniedError') {
                console.log('Mic permissions denied');
            }

            navigator.permissions.query(
                { name: 'microphone' }
            ).then(function(permissionStatus){
                if (permissionStatus.state === "denied") {
                    setMicButtonState("blocked");
                }
            })
        })
    }

    function setMicMode() {
        const xPathModeButton = document.evaluate ('//div[@id = "microphone"]//button[contains(@aria-label, "Record audio")]', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
        xPathModeButton?.singleNodeValue?.click();
        console.log("Mic mode set clicked ",xPathModeButton?.singleNodeValue? true : false);
    }
 
    let listeningInterval = null;
    let micOnTimeout = null;
    function toggleRecord() {
        clearInterval(listeningInterval)
        clearTimeout(micOnTimeout)
        textBox.value = "";
        if (recordingState === "off") {
            // Clear and start recording
            console.log("start recording");
            const xPathResultClearBtn = document.evaluate ('//div[@id = "microphone"]//button[contains(@title, "Clear")] | //div[@id = "microphone"]//button[contains(@class, "svelte-p87ime")]', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
            if (xPathResultClearBtn) {
                xPathResultClearBtn?.singleNodeValue?.click();
                console.log("Clicked clear button ", xPathResultClearBtn?.singleNodeValue ? true : false);
            }
            setMicMode();
            micOnTimeout = setTimeout(() => {
                setMicButtonState("on");
                const xPathResultRecordBtn = document.evaluate ('//div[@id = "microphone"]//button[contains(@class, "record")]', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                xPathResultRecordBtn?.singleNodeValue?.click();
                console.log("start record clicked ",xPathResultRecordBtn?.singleNodeValue? true : false);
                const RecordingString = "Recording... 0:"
                let counter = 30;
                textBox.value = RecordingString + counter.toLocaleString('en-US', {minimumIntegerDigits: 2, useGrouping:false})
                listeningInterval = setInterval(() => {
                    counter-=1;
                    textBox.value = RecordingString + counter.toLocaleString('en-US', {minimumIntegerDigits: 2, useGrouping:false})
                    if (counter === 0) {
                        toggleRecord();
                    }
                }, 1000)

            }, 500);
        } else {
            // Stop recording
            console.log("stop recording");
            const xPathResultStopBtn = document.evaluate ('//div[@id = "microphone"]//button[contains(@class, "stop-button")]', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
            xPathResultStopBtn?.singleNodeValue?.click();
            console.log("stop record clicked ",xPathResultStopBtn?.singleNodeValue? true : false);
            setMicButtonState("off");
        }
    }

    function setMicButtonState(state) {
        if (micComponent) {
            micInitButton.style.display = "none"
            micOffButton.style.display = "none";
            micOnButton.style.display = "none";
            micBlockedButton.style.display = "none";
            chatBoxGroup.classList.remove("mic-listen-border");
            if (state === "init") {
                micInitButton.style.display = "flex";
            } else if (state === "off") {
                micOffButton.style.display = "flex";
            } else if (state === "on") {
                micOnButton.style.display = "flex";
                chatBoxGroup.classList.add("mic-listen-border");
            } else if (state === "blocked") {
                micBlockedButton.style.display = "flex";
            }
            recordingState = state;
        }
        
    }
    
}