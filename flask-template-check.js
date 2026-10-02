
        // Primary Web UI Logic Flow
        const startBtn = document.getElementById('btn-start');
        const stopBtn = document.getElementById('btn-stop');
        const voiceBtn = document.getElementById('btn-voice');
        const vibrateBtn = document.getElementById('btn-vibrate');
        const statusDisplay = document.getElementById('status-display');
        const devToggleBtn = document.getElementById('btn-dev-toggle');
        const devPanel = document.getElementById('dev-panel');
        const devHudImg = document.getElementById('dev-hud-img');
        const devFpsStat = document.getElementById('dev-fps-stat');
        const devThreatStat = document.getElementById('dev-threat-stat');
        const videoElement = document.getElementById('video-stream');
        const canvasElement = document.getElementById('frame-canvas');
        const canvasCtx = canvasElement.getContext('2d');

        // Application State variables
        let isAssistanceActive = false;
        let isVoiceEnabled = true;
        let isVibrationEnabled = true;
        let isDevModeActive = false;
        let streamTrack = null;
        let processLoopInterval = null;
        let speechSynth = window.speechSynthesis;
        const languageSelect = document.getElementById('language-select');
        let guidanceLanguage = localStorage.getItem('safepath-language') || 'en';
        let lastSpokenText = "";
        let lastSpokenTime = 0;

        const UI_TEXT = {
            en: { language: 'Guidance language', assistanceStatus: 'Assistance Status', start: 'Start Assistance', stop: 'Stop Assistance', voiceOn: 'Voice ON', voiceOff: 'Voice OFF', vibrationOn: 'Vibrate ON', vibrationOff: 'Vibrate OFF', guide: '🎙️ Voice Launch Guide', guideText: 'You can launch SafePath completely hands-free using Gemini, Google Assistant, or Siri:', step1: 'Install this app on your homescreen using the browser menu (Add to Home Screen).', step2: 'Say: "Hey Google, open SafePath" (or Siri on Apple devices).', step3: 'To start scanning automatically upon launch, link to: ?action=start', developer: 'Developer Settings', inactive: 'INACTIVE', pathClear: 'PATH CLEAR', obstacle: 'OBSTACLE DETECTED', cameraError: 'CAMERA ERROR', connecting: 'CONNECTING...', connectingCamera: 'CONNECTING CAMERA...', cameraFailure: 'Failed to connect to camera.', started: 'Navigation assistance started.', stopped: 'Navigation assistance stopped.', voiceEnabled: 'Voice warnings enabled.' },
            hi: { language: 'मार्गदर्शन की भाषा', assistanceStatus: 'सहायता की स्थिति', start: 'सहायता शुरू करें', stop: 'सहायता रोकें', voiceOn: 'आवाज़ चालू', voiceOff: 'आवाज़ बंद', vibrationOn: 'कंपन चालू', vibrationOff: 'कंपन बंद', guide: '🎙️ आवाज़ से शुरू करने की जानकारी', guideText: 'Gemini, Google Assistant या Siri से SafePath शुरू करें:', step1: 'ब्राउज़र मेनू से इस ऐप को होम स्क्रीन पर जोड़ें।', step2: 'कहें: "Hey Google, open SafePath" (या Apple डिवाइस पर Siri से कहें)।', step3: 'ऐप खुलते ही स्कैन शुरू करने के लिए लिंक में ?action=start जोड़ें।', developer: 'डेवलपर सेटिंग', inactive: 'निष्क्रिय', pathClear: 'आगे का रास्ता साफ़ दिख रहा है', obstacle: 'रास्ते में बाधा मिली', cameraError: 'कैमरा त्रुटि', connecting: 'कनेक्ट हो रहा है...', connectingCamera: 'कैमरा कनेक्ट हो रहा है...', cameraFailure: 'कैमरे से कनेक्ट नहीं हो सका।', started: 'नेविगेशन सहायता शुरू हुई।', stopped: 'नेविगेशन सहायता बंद हुई।', voiceEnabled: 'आवाज़ की चेतावनियाँ चालू हैं।' },
            te: { language: 'మార్గదర్శక భాష', assistanceStatus: 'సహాయ స్థితి', start: 'సహాయాన్ని ప్రారంభించండి', stop: 'సహాయాన్ని ఆపండి', voiceOn: 'వాయిస్ ఆన్', voiceOff: 'వాయిస్ ఆఫ్', vibrationOn: 'వైబ్రేషన్ ఆన్', vibrationOff: 'వైబ్రేషన్ ఆఫ్', guide: '🎙️ వాయిస్‌తో ప్రారంభించే మార్గదర్శకం', guideText: 'Gemini, Google Assistant లేదా Siri ద్వారా SafePath ప్రారంభించవచ్చు:', step1: 'బ్రౌజర్ మెనూ ద్వారా ఈ యాప్‌ను హోమ్ స్క్రీన్‌కు జోడించండి.', step2: '"Hey Google, open SafePath" అని చెప్పండి (Apple పరికరాల్లో Siriని ఉపయోగించండి).', step3: 'యాప్ తెరవగానే స్కానింగ్ ప్రారంభించడానికి లింక్‌కు ?action=start జోడించండి.', developer: 'డెవలపర్ సెట్టింగ్‌లు', inactive: 'నిష్క్రియం', pathClear: 'ముందు దారి ఖాళీగా కనిపిస్తోంది', obstacle: 'దారిలో అడ్డంకి గుర్తించబడింది', cameraError: 'కెమెరా లోపం', connecting: 'కనెక్ట్ అవుతోంది...', connectingCamera: 'కెమెరాకు కనెక్ట్ అవుతోంది...', cameraFailure: 'కెమెరాకు కనెక్ట్ కాలేదు.', started: 'నావిగేషన్ సహాయం ప్రారంభమైంది.', stopped: 'నావిగేషన్ సహాయం ఆపబడింది.', voiceEnabled: 'వాయిస్ హెచ్చరికలు ప్రారంభించబడ్డాయి.' }
        };
        const words = {
            en: { Person: 'person', Bicycle: 'bicycle', Car: 'car', Motorcycle: 'motorcycle', Bus: 'bus', Truck: 'truck', 'Traffic Light': 'traffic light', 'Stop Sign': 'stop sign', Backpack: 'backpack', Umbrella: 'umbrella', Handbag: 'handbag', Default: 'obstacle', CENTER: 'ahead', LEFT: 'to your left', RIGHT: 'to your right', approaching: 'approaching', receding: 'moving away', stable: 'nearby' },
            hi: { Person: 'व्यक्ति', Bicycle: 'साइकिल', Car: 'कार', Motorcycle: 'मोटरसाइकिल', Bus: 'बस', Truck: 'ट्रक', 'Traffic Light': 'ट्रैफिक लाइट', 'Stop Sign': 'रुकने का संकेत', Backpack: 'बैग', Umbrella: 'छाता', Handbag: 'हैंडबैग', Default: 'बाधा', CENTER: 'सामने', LEFT: 'आपकी बाईं ओर', RIGHT: 'आपकी दाईं ओर', approaching: 'पास आ रहा है', receding: 'दूर जा रहा है', stable: 'पास है' },
            te: { Person: 'వ్యక్తి', Bicycle: 'సైకిల్', Car: 'కారు', Motorcycle: 'మోటార్ సైకిల్', Bus: 'బస్సు', Truck: 'ట్రక్', 'Traffic Light': 'ట్రాఫిక్ లైట్', 'Stop Sign': 'ఆపు గుర్తు', Backpack: 'బ్యాగ్', Umbrella: 'గొడుగు', Handbag: 'హ్యాండ్‌బ్యాగ్', Default: 'అడ్డంకి', CENTER: 'ముందు', LEFT: 'మీ ఎడమవైపు', RIGHT: 'మీ కుడివైపు', approaching: 'దగ్గరకు వస్తోంది', receding: 'దూరంగా వెళ్తోంది', stable: 'దగ్గరలో ఉంది' }
        };
        if (!UI_TEXT[guidanceLanguage]) guidanceLanguage = 'en';
        function applyLanguage() {
            const t = UI_TEXT[guidanceLanguage] || UI_TEXT.en;
            const set = (selector, value) => { const el = document.querySelector(selector); if (el) el.textContent = value; };
            document.documentElement.lang = ({ en: 'en', hi: 'hi', te: 'te' })[guidanceLanguage] || 'en';
            set('#language-label', t.language); set('.status-title', t.assistanceStatus);
            set('#btn-start', t.start); set('#btn-stop', t.stop);
            set('#btn-voice', isVoiceEnabled ? t.voiceOn : t.voiceOff);
            set('#btn-vibrate', isVibrationEnabled ? t.vibrationOn : t.vibrationOff);
            set('.assistant-guide-title', t.guide); set('.assistant-guide > p', t.guideText);
            const steps = document.querySelectorAll('.assistant-guide-step li');
            [t.step1, t.step2, t.step3].forEach((value, i) => { if (steps[i]) steps[i].innerHTML = value.replace('Add to Home Screen', '<strong>Add to Home Screen</strong>').replace('"Hey Google, open SafePath"', '<strong class="assistant-code">"Hey Google, open SafePath"</strong>').replace('?action=start', '<strong class="assistant-code">?action=start</strong>'); });
            set('#dev-toggle-label', t.developer);
            if (!isAssistanceActive && statusDisplay.textContent.trim() === 'INACTIVE') statusDisplay.textContent = t.inactive;
        }
        languageSelect.value = guidanceLanguage;
        languageSelect.addEventListener('change', () => { guidanceLanguage = languageSelect.value; localStorage.setItem('safepath-language', guidanceLanguage); lastSpokenText = ''; applyLanguage(); });

        function localizedWarning(item) {
            if (guidanceLanguage === 'en' || !item) return item ? null : (UI_TEXT[guidanceLanguage] || UI_TEXT.en).pathClear;
            const t = words[guidanceLanguage] || words.en;
            const name = t[item.class_name] || item.class_name;
            const direction = t[item.direction] || t.CENTER;
            const motion = item.motion === 'approaching' ? t.approaching : item.motion === 'receding' ? t.receding : t.stable;
            if (guidanceLanguage === 'hi') return item.motion === 'approaching' ? `${direction} ${name} ${motion}।` : `${direction} ${name} है।`;
            return item.motion === 'approaching' ? `${direction} ${name} ${motion}.` : `${direction} ${name} ఉంది.`;
        }

        applyLanguage();

        // Sound Synthesis helper
        function speak(text) {
            if (!isVoiceEnabled || !speechSynth) return;
            
            const now = Date.now();
            // Local fallback filter to prevent immediate repeat speech
            if (text === lastSpokenText && (now - lastSpokenTime < 3000)) {
                return;
            }
            
            // Cancel active speak tasks to allow fast immediate warning overrides
            speechSynth.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            const speechLocale = ({ en: 'en-US', hi: 'hi-IN', te: 'te-IN' })[guidanceLanguage] || 'en-US';
            utterance.lang = speechLocale;
            const voices = speechSynth.getVoices();
            const matchingVoice = voices.find(v => v.lang.toLowerCase() === speechLocale.toLowerCase()) || voices.find(v => v.lang.toLowerCase().startsWith(speechLocale.slice(0, 2)));
            if (matchingVoice) utterance.voice = matchingVoice;
            utterance.rate = 1.1; // slightly faster for utility speed
            utterance.volume = 1.0;
            
            speechSynth.speak(utterance);
            lastSpokenText = text;
            lastSpokenTime = now;
        }

        // Tactile Haptic Vibration helper
        function triggerHapticFeedback(pattern) {
            if (isVibrationEnabled && navigator.vibrate) {
                navigator.vibrate(pattern);
            }
        }

        // Initialize Camera Stream
        async function startCamera() {
            try {
                // Request environment/back camera specifically for orientation walking
                const constraints = {
                    video: {
                        facingMode: { ideal: 'environment' },
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    },
                    audio: false
                };
                
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                videoElement.srcObject = stream;
                streamTrack = stream.getVideoTracks()[0];
                
                // Configure canvas sizes
                canvasElement.width = 640;
                canvasElement.height = 480;
                
                return true;
            } catch (err) {
                console.error("Camera connection failed: ", err);
                speak((UI_TEXT[guidanceLanguage] || UI_TEXT.en).cameraFailure);
                statusDisplay.innerText = (UI_TEXT[guidanceLanguage] || UI_TEXT.en).cameraError;
                statusDisplay.className = "status-value status-detected";
                return false;
            }
        }

        // Capture canvas and send to backend
        async function sendFrameToServer() {
            if (!isAssistanceActive) return;

            // Draw current video frame onto the hidden canvas
            try {
                canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                // Convert to compressed base64 JPEG format
                const base64Image = canvasElement.toDataURL('image/jpeg', 0.6);
                
                const startCall = Date.now();
                const response = await fetch('/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: base64Image,
                        dev_mode: isDevModeActive
                    })
                });

                if (!response.ok) {
                    throw new Error("Server HTTP error: " + response.status);
                }

                const data = await response.json();
                const callLatency = Date.now() - startCall;
                
                if (isDevModeActive) {
                    devFpsStat.innerText = `Processing Delay: ${callLatency}ms`;
                }
                
                handlePipelineResults(data);
            } catch (err) {
                console.error("Processing frame error: ", err);
                // Handle offline or disconnect states
                statusDisplay.innerText = "CONNECTING...";
                statusDisplay.className = "status-value";
                statusDisplay.style.color = "#FFCC00";
            }
        }

        // Handle pipeline risk results
        function handlePipelineResults(data) {
            if (!isAssistanceActive) return;

            // 1. Update text display
            const ui = UI_TEXT[guidanceLanguage] || UI_TEXT.en;
            statusDisplay.innerText = data.status === 'PATH CLEAR' ? ui.pathClear : data.status === 'OBSTACLE DETECTED' ? ui.obstacle : data.status;
            
            if (data.status === "OBSTACLE DETECTED") {
                statusDisplay.className = "status-value status-detected";
                // Trigger quick warning double-pulse vibration
                triggerHapticFeedback([150, 100, 150]);
            } else {
                statusDisplay.className = "status-value status-clear";
            }

            // 2. Play warning voice audio
            if (data.spoken_warning) {
                speak(localizedWarning(data.highest_risk) || data.spoken_warning);
            }

            // 3. Render optional developer mode metrics
            if (isDevModeActive) {
                if (data.debug_image) {
                    devHudImg.src = data.debug_image;
                }
                
                const highest = data.highest_risk;
                if (highest) {
                    devThreatStat.innerText = `Target: ${highest.class_name} (${highest.direction}) - Risk: ${Math.round(highest.risk_score)}% [${highest.motion}]`;
                } else {
                    devThreatStat.innerText = "Target: None";
                }
            }
        }

        // Toggle Assistance State
        async function startAssistance() {
            statusDisplay.innerText = (UI_TEXT[guidanceLanguage] || UI_TEXT.en).connectingCamera;
            statusDisplay.className = "status-value";
            statusDisplay.style.color = "#FFCC00";
            
            const cameraReady = await startCamera();
            if (!cameraReady) return;
            
            isAssistanceActive = true;
            
            // Update UI targets
            startBtn.style.display = 'none';
            stopBtn.style.display = 'flex';
            stopBtn.focus();
            
            statusDisplay.innerText = (UI_TEXT[guidanceLanguage] || UI_TEXT.en).pathClear;
            statusDisplay.className = "status-value status-clear";
            
            speak((UI_TEXT[guidanceLanguage] || UI_TEXT.en).started);
            triggerHapticFeedback([100, 50, 100]);
            
            // Start frame process loop at ~5 FPS (every 220ms) to ensure low-latency responsiveness
            processLoopInterval = setInterval(sendFrameToServer, 220);
        }

        function stopAssistance() {
            isAssistanceActive = false;
            
            // Cancel loop
            if (processLoopInterval) {
                clearInterval(processLoopInterval);
                processLoopInterval = null;
            }
            
            // Stop camera hardware tracks to release battery usage
            if (streamTrack) {
                streamTrack.stop();
                streamTrack = null;
            }
            
            // Clear developer HUD
            devHudImg.src = "";
            devThreatStat.innerText = "Target: None";
            
            // Update UI
            stopBtn.style.display = 'none';
            startBtn.style.display = 'flex';
            startBtn.focus();
            
            statusDisplay.innerText = (UI_TEXT[guidanceLanguage] || UI_TEXT.en).inactive;
            statusDisplay.className = "status-value";
            statusDisplay.style.color = "#666666";
            
            speak((UI_TEXT[guidanceLanguage] || UI_TEXT.en).stopped);
            triggerHapticFeedback([300]);
        }

        // Toggle Controls Functions
        function toggleVoice() {
            isVoiceEnabled = !isVoiceEnabled;
            if (isVoiceEnabled) {
                voiceBtn.classList.add('active');
                voiceBtn.innerText = (UI_TEXT[guidanceLanguage] || UI_TEXT.en).voiceOn;
                voiceBtn.setAttribute('aria-pressed', 'true');
                voiceBtn.setAttribute('aria-label', "Toggle Voice Warnings, currently ON.");
                speak((UI_TEXT[guidanceLanguage] || UI_TEXT.en).voiceEnabled);
            } else {
                voiceBtn.classList.remove('active');
                voiceBtn.innerText = (UI_TEXT[guidanceLanguage] || UI_TEXT.en).voiceOff;
                voiceBtn.setAttribute('aria-pressed', 'false');
                voiceBtn.setAttribute('aria-label', "Toggle Voice Warnings, currently OFF.");
                if (speechSynth) speechSynth.cancel();
            }
            triggerHapticFeedback(100);
        }

        function toggleVibration() {
            isVibrationEnabled = !isVibrationEnabled;
            if (isVibrationEnabled) {
                vibrateBtn.classList.add('active');
                vibrateBtn.innerText = (UI_TEXT[guidanceLanguage] || UI_TEXT.en).vibrationOn;
                vibrateBtn.setAttribute('aria-pressed', 'true');
                vibrateBtn.setAttribute('aria-label', "Toggle Phone Vibration Alerts, currently ON.");
                triggerHapticFeedback([100, 50, 100]);
            } else {
                vibrateBtn.classList.remove('active');
                vibrateBtn.innerText = (UI_TEXT[guidanceLanguage] || UI_TEXT.en).vibrationOff;
                vibrateBtn.setAttribute('aria-pressed', 'false');
                vibrateBtn.setAttribute('aria-label', "Toggle Phone Vibration Alerts, currently OFF.");
            }
        }

        // Toggle Developer Panel
        function toggleDevPanel() {
            isDevModeActive = !isDevModeActive;
            if (isDevModeActive) {
                devPanel.classList.add('active');
                devToggleBtn.setAttribute('aria-expanded', 'true');
                devToggleBtn.style.color = 'var(--text-color)';
            } else {
                devPanel.classList.remove('active');
                devToggleBtn.setAttribute('aria-expanded', 'false');
                devToggleBtn.style.color = '#666666';
                devHudImg.src = "";
            }
            triggerHapticFeedback(50);
        }

        // Bind interactive events
        startBtn.addEventListener('click', startAssistance);
        stopBtn.addEventListener('click', stopAssistance);
        voiceBtn.addEventListener('click', toggleVoice);
        vibrateBtn.addEventListener('click', toggleVibration);
        devToggleBtn.addEventListener('click', toggleDevPanel);

        // Global Keyboard Hotkeys for screen-reader navigation
        window.addEventListener('keydown', (e) => {
            const key = e.key.toLowerCase();
            // Space bar triggers start/stop toggle if not focusing a button already
            if (key === ' ' || key === 'enter') {
                const active = document.activeElement;
                if (active.tagName !== 'BUTTON') {
                    e.preventDefault(); // prevent scrolling page down on spacebar
                    if (isAssistanceActive) {
                        stopAssistance();
                    } else {
                        startAssistance();
                    }
                }
            }
            // Mute hotkey
            else if (key === 'm' || key === 'v') {
                if (key === 'm') toggleVoice();
                if (key === 'v') toggleVibration();
            }
        });

        // Parse query string parameters on window load to support Voice Assistant auto-start
        window.addEventListener('load', () => {
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.get('action') === 'start') {
                setTimeout(() => {
                    startAssistance();
                }, 800); // 800ms delay to satisfy browser user gesture context rules smoothly
            }
        });

        // Register PWA Service Worker for offline installs
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('/sw.js')
                    .then(reg => console.log('PWA Service Worker registered successfully', reg.scope))
                    .catch(err => console.error('PWA Service Worker registration failed', err));
            });
        }
    