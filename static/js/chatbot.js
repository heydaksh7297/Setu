/**
 * ============================================================
 *  JECRC Foundation - AI Helpdesk Chatbot
 *  🔥 Voice + Text Format Choice + Rich Media + Links
 *  Project: J-TECHTRIX 7.0
 * ============================================================
 */

const API_URL = '/api/chat';
const VOICE_API_URL = '/api/voice-chat';
const TTS_API_URL = '/api/text-to-speech';
const SESSION_ID = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const typingIndicator = document.getElementById('typingIndicator');
const charCount = document.getElementById('charCount');

// Voice State
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordingStream = null;

// ══════════════════════════════════════
// 🔧 UTILITIES
// ══════════════════════════════════════

function getCurrentTime() {
    const now = new Date();
    let hours = now.getHours();
    const mins = now.getMinutes().toString().padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    return `${hours}:${mins} ${ampm}`;
}

function escapeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBotText(text) {
    if (!text) return '<p>Sorry, couldn\'t process that.</p>';
    let f = escapeHTML(text);
    f = f.replace(/\*\*(.*?)\*\*/g, '<strong>\$1</strong>');
    f = f.replace(/__(.*?)__/g, '<strong>\$1</strong>');
    const lines = f.split('\n').filter(l => l.trim());
    let html = '', inUL = false, inOL = false;
    lines.forEach(line => {
        const t = line.trim();
        if (t.startsWith('• ') || t.startsWith('- ') || t.startsWith('* ')) {
            if (inOL) { html += '</ol>'; inOL = false; }
            if (!inUL) { html += '<ul>'; inUL = true; }
            html += `<li>${t.substring(2)}</li>`;
        } else if (/^\d+[\.\)]\s/.test(t)) {
            if (inUL) { html += '</ul>'; inUL = false; }
            if (!inOL) { html += '<ol>'; inOL = true; }
            html += `<li>${t.replace(/^\d+[\.\)]\s/, '')}</li>`;
        } else {
            if (inUL) { html += '</ul>'; inUL = false; }
            if (inOL) { html += '</ol>'; inOL = false; }
            html += `<p>${t}</p>`;
        }
    });
    if (inUL) html += '</ul>';
    if (inOL) html += '</ol>';
    return html || `<p>${f}</p>`;
}

function scrollToBottom() {
    if (chatMessages) {
        setTimeout(() => {
            chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
        }, 100);
    }
}

function showTyping() {
    if (typingIndicator) { typingIndicator.classList.add('show'); scrollToBottom(); }
}

function hideTyping() {
    if (typingIndicator) typingIndicator.classList.remove('show');
}

function setInputState(enabled) {
    if (userInput) userInput.disabled = !enabled;
    if (sendBtn) sendBtn.disabled = !enabled;
}

function updateCharCount() {
    if (charCount && userInput) {
        const len = userInput.value.length;
        charCount.textContent = `${len}/500`;
        charCount.style.color = len > 450 ? '#ef4444' : len > 350 ? '#f59e0b' : '#64748b';
    }
}

// ══════════════════════════════════════
// 🔗 NEW: RENDER WEBSITE LINKS
// ══════════════════════════════════════

function renderLinks(links) {
    if (!links || links.length === 0) return '';

    let html = '<div class="bot-links">';
    html += '<div class="links-header"><i class="fas fa-external-link-alt"></i> Helpful Links:</div>';
    html += '<div class="links-container">';

    links.forEach(link => {
        const icon = link.icon || '🔗';
        const text = escapeHTML(link.text || 'Visit Link');
        const url = link.url || '#';

        html += `
            <a href="${url}" target="_blank" rel="noopener noreferrer" class="bot-link-card" title="${text}">
                <span class="link-icon">${icon}</span>
                <span class="link-text">${text}</span>
                <i class="fas fa-arrow-right link-arrow"></i>
            </a>
        `;
    });

    html += '</div></div>';
    return html;
}

// ══════════════════════════════════════
// 📸 NEW: RENDER RICH MEDIA
// ══════════════════════════════════════

function renderMedia(media) {
    if (!media) return '';

    let html = '<div class="bot-media">';

    switch (media.type) {
        case 'map':
            html += renderMapMedia(media);
            break;
        case 'gallery':
            html += renderGalleryMedia(media);
            break;
        case 'card':
            html += renderCardMedia(media);
            break;
        default:
            return '';
    }

    html += '</div>';
    return html;
}

function renderMapMedia(media) {
    const title = escapeHTML(media.title || 'Location');
    const mapsLink = media.maps_link || media.embed_url || '#';

    return `
        <div class="media-map">
            <div class="map-preview">
                <div class="map-placeholder">
                    <i class="fas fa-map-marked-alt"></i>
                    <span>${title}</span>
                </div>
                <a href="${mapsLink}" target="_blank" rel="noopener noreferrer" class="map-open-btn">
                    <i class="fas fa-external-link-alt"></i> Open in Google Maps
                </a>
            </div>
        </div>
    `;
}

function renderGalleryMedia(media) {
    const title = escapeHTML(media.title || 'Gallery');
    const images = media.images || [];

    if (images.length === 0) return '';

    let html = `<div class="media-gallery">`;
    html += `<div class="gallery-title"><i class="fas fa-images"></i> ${title}</div>`;
    html += `<div class="gallery-grid">`;

    images.forEach(img => {
        const caption = escapeHTML(img.caption || '');
        const url = img.url || '';

        html += `
            <div class="gallery-item" onclick="openImagePreview('${url}', '${caption}')">
                <div class="gallery-img-wrapper">
                    <img src="${url}" alt="${caption}" loading="lazy" 
                         onerror="this.parentElement.innerHTML='<div class=\\'gallery-placeholder\\'><i class=\\'fas fa-image\\'></i><span>${caption}</span></div>'"
                    />
                </div>
                ${caption ? `<div class="gallery-caption">${caption}</div>` : ''}
            </div>
        `;
    });

    html += `</div></div>`;
    return html;
}

function renderCardMedia(media) {
    const title = escapeHTML(media.title || '');
    const subtitle = escapeHTML(media.subtitle || '');
    const image = media.image || '';
    const website = media.website || '';

    let html = `<div class="media-card">`;

    if (image) {
        html += `
            <div class="card-image">
                <img src="${image}" alt="${title}" loading="lazy"
                     onerror="this.parentElement.innerHTML='<div class=\\'card-img-placeholder\\'><i class=\\'fas fa-building\\'></i></div>'"
                />
            </div>
        `;
    }

    html += `<div class="card-body">`;
    if (title) html += `<div class="card-title">${title}</div>`;
    if (subtitle) html += `<div class="card-subtitle">${subtitle}</div>`;

    if (website) {
        html += `
            <a href="${website}" target="_blank" rel="noopener noreferrer" class="card-link">
                <i class="fas fa-globe"></i> Visit Website
            </a>
        `;
    }

    html += `</div></div>`;
    return html;
}

// ══════════════════════════════════════
// 📸 Image Preview Modal
// ══════════════════════════════════════

function openImagePreview(url, caption) {
    if (!url) return;

    // Remove existing modal
    const existing = document.getElementById('imagePreviewModal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'imagePreviewModal';
    modal.className = 'image-preview-modal';
    modal.innerHTML = `
        <div class="image-preview-overlay" onclick="closeImagePreview()"></div>
        <div class="image-preview-content">
            <button class="image-preview-close" onclick="closeImagePreview()">
                <i class="fas fa-times"></i>
            </button>
            <img src="${url}" alt="${escapeHTML(caption)}" />
            ${caption ? `<div class="image-preview-caption">${escapeHTML(caption)}</div>` : ''}
        </div>
    `;

    document.body.appendChild(modal);
    setTimeout(() => modal.classList.add('show'), 10);
}

function closeImagePreview() {
    const modal = document.getElementById('imagePreviewModal');
    if (modal) {
        modal.classList.remove('show');
        setTimeout(() => modal.remove(), 300);
    }
}

// ══════════════════════════════════════
// 💬 ADD MESSAGE (UPDATED with links & media)
// ══════════════════════════════════════

function addMessage(text, sender, extra = {}) {
    const div = document.createElement('div');
    div.className = `message ${sender}-message`;
    const icon = sender === 'bot' ? 'fa-robot' : 'fa-user';
    const time = getCurrentTime();
    let bubble = sender === 'bot' ? formatBotText(text) : `<p>${escapeHTML(text)}</p>`;

    // 🔗 Add links if present (bot only)
    let linksHTML = '';
    if (sender === 'bot' && extra.links && extra.links.length > 0) {
        linksHTML = renderLinks(extra.links);
    }

    // 📸 Add media if present (bot only)
    let mediaHTML = '';
    if (sender === 'bot' && extra.media) {
        mediaHTML = renderMedia(extra.media);
    }

    // Debug info
    let debug = '';
    if (extra.debugInfo && sender === 'bot') {
        debug = `<div class="msg-debug">🧠 ${extra.debugInfo.category || ''} | ${extra.debugInfo.confidence || ''}</div>`;
    }

    // Feedback buttons
    let feedback = '';
    if (sender === 'bot') {
        const id = 'msg_' + Date.now();
        feedback = `<div class="msg-meta"><span class="msg-time">${time}</span>
            <div class="feedback-btns">
                <button class="fb-btn" onclick="giveFeedback(this,'${id}','up')"><i class="fas fa-thumbs-up"></i></button>
                <button class="fb-btn" onclick="giveFeedback(this,'${id}','down')"><i class="fas fa-thumbs-down"></i></button>
            </div></div>`;
    }

    div.innerHTML = `
        <div class="msg-avatar"><i class="fas ${icon}"></i></div>
        <div class="msg-content">
            <div class="msg-bubble">${bubble}</div>
            ${mediaHTML}
            ${linksHTML}
            ${debug}
            ${sender === 'bot' ? feedback : `<span class="msg-time">${time}</span>`}
        </div>
    `;

    chatMessages.appendChild(div);
    scrollToBottom();
}

// ══════════════════════════════════════
// 🔊 PLAY AUDIO RESPONSE
// ══════════════════════════════════════

function playAudioResponse(url) {
    if (!url) return;
    try {
        console.log('🔊 Playing:', url);
        const audio = new Audio(url);
        audio.volume = 0.8;
        audio.play().catch((err) => {
            console.warn('⚠️ Autoplay blocked:', err);
        });
    } catch (e) {
        console.error('Audio error:', e);
    }
}

// ══════════════════════════════════════
// 🔥 FORMAT CHOICE SYSTEM
// ══════════════════════════════════════

function showFormatChoice(source, payload) {
    const existing = document.getElementById('formatChoice');
    if (existing) existing.remove();

    const choiceDiv = document.createElement('div');
    choiceDiv.className = 'message bot-message';
    choiceDiv.id = 'formatChoice';

    const questionText = source === 'voice'
        ? '🎤 Voice recorded! How would you like the answer?'
        : '💬 How would you like the answer?';

    choiceDiv.innerHTML = `
        <div class="msg-avatar"><i class="fas fa-robot"></i></div>
        <div class="msg-content">
            <div class="msg-bubble">
                <p>${questionText}</p>
                <div class="format-choice-btns">
                    <button class="format-btn format-text" onclick="handleFormatChoice('text')">
                        <i class="fas fa-font"></i> Text Only
                    </button>
                    <button class="format-btn format-speech" onclick="handleFormatChoice('speech')">
                        <i class="fas fa-volume-up"></i> Speech Only
                    </button>
                    <button class="format-btn format-both" onclick="handleFormatChoice('both')">
                        <i class="fas fa-magic"></i> Both
                    </button>
                </div>
            </div>
            <span class="msg-time">${getCurrentTime()}</span>
        </div>
    `;

    chatMessages.appendChild(choiceDiv);
    scrollToBottom();

    window._pendingSource = source;
    window._pendingPayload = payload;
}

function handleFormatChoice(format) {
    const source = window._pendingSource;
    const payload = window._pendingPayload;

    if (!source || !payload) return;

    const choiceEl = document.getElementById('formatChoice');
    if (choiceEl) choiceEl.remove();

    window._pendingSource = null;
    window._pendingPayload = null;

    if (source === 'text') {
        sendTextWithFormat(payload, format);
    } else if (source === 'voice') {
        const formatLabels = { text: '📝 Text Only', speech: '🔊 Speech Only', both: '📝🔊 Both' };
        addMessage(`🎤 Voice sent (${formatLabels[format]})`, 'user');
        sendVoiceToServer(payload, format);
    }
}

// ══════════════════════════════════════
// 📤 TEXT INPUT → FORMAT CHOICE → SEND
// ══════════════════════════════════════

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) {
        const w = document.querySelector('.input-wrapper');
        w.style.animation = 'shake 0.5s';
        setTimeout(() => { w.style.animation = ''; }, 500);
        return;
    }

    addMessage(message, 'user');
    userInput.value = '';
    updateCharCount();

    showFormatChoice('text', message);
}

/**
 * 🎯 Send text message with chosen format (UPDATED with links & media)
 */
async function sendTextWithFormat(message, format) {
    setInputState(false);
    showTyping();

    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                session_id: SESSION_ID
            })
        });

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }

        const data = await res.json();
        hideTyping();

        const reply = data.reply || "Sorry, couldn't understand.";
        const links = data.links || [];
        const media = data.media || null;

        // Show text reply if needed (with links & media)
        if (format === 'text' || format === 'both') {
            addMessage(reply, 'bot', { links, media });
        }

        // Generate + play speech if needed
        if (format === 'speech' || format === 'both') {

            if (format === 'speech') {
                // For speech-only, show links & media but minimal text
                addMessage("🔊 Playing audio response...", 'bot', {
                    links,
                    media
                });
            }

            try {
                const ttsRes = await fetch(TTS_API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: reply,
                        language: 'hi'
                    })
                });

                if (ttsRes.ok) {
                    const ttsData = await ttsRes.json();

                    if (ttsData.audio_url) {
                        playAudioResponse(ttsData.audio_url);
                    } else {
                        console.warn('No audio_url in TTS response');

                        if (format === 'speech') {
                            addMessage(reply, 'bot');
                        }
                    }
                } else {
                    console.warn('TTS API error:', ttsRes.status);

                    if (format === 'speech') {
                        addMessage(
                            "⚠️ Audio generation failed. Here's the text:",
                            'bot'
                        );
                        addMessage(reply, 'bot', { links, media });
                    }
                }
            } catch (ttsErr) {
                console.warn('TTS error:', ttsErr);

                if (format === 'speech') {
                    addMessage(
                        "⚠️ Audio failed. Here's the text instead:",
                        'bot'
                    );
                    addMessage(reply, 'bot', { links, media });
                }
            }
        }

    } catch (e) {
        hideTyping();
        console.error('Chat error:', e);
        addMessage("⚠️ Something went wrong. Try again.", 'bot');

    } finally {
        setInputState(true);
        userInput.focus();
    }
}
// ══════════════════════════════════════
// 🚀 QUICK TOPICS
// ══════════════════════════════════════

function sendQuick(msg) {
    userInput.value = msg;
    sendMessage();
    if (window.innerWidth <= 768) toggleSidebar();
}

function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

// ══════════════════════════════════════
// 👍 FEEDBACK
// ══════════════════════════════════════

function giveFeedback(btn, id, type) {
    btn.closest('.feedback-btns').querySelectorAll('.fb-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    fetch('/api/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message_id: id, feedback: type, session_id: SESSION_ID })
    }).catch(() => {});
}

// ══════════════════════════════════════
// 🗑️ CLEAR + SIDEBAR
// ══════════════════════════════════════

function clearChat() {
    if (!confirm('🗑️ Clear all messages?')) return;
    chatMessages.querySelectorAll('.message').forEach((m, i) => { if (i > 0) m.remove(); });
    scrollToBottom();
}

function toggleSidebar() {
    const s = document.getElementById('sidebar');
    const o = document.getElementById('sidebarOverlay');
    if (s && o) { s.classList.toggle('open'); o.classList.toggle('show'); }
}

// ══════════════════════════════════════
// 🎤 VOICE RECORDING SYSTEM
// ══════════════════════════════════════

async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('❌ Use Google Chrome for voice feature.');
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: 44100, channelCount: 1 }
        });
        recordingStream = stream;
        let mimeType = 'audio/webm';
        for (const t of ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']) {
            if (MediaRecorder.isTypeSupported(t)) { mimeType = t; break; }
        }
        mediaRecorder = new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 128000 });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            if (audioChunks.length === 0) {
                addMessage("🎤 No audio captured.", 'bot');
                cleanupStream();
                return;
            }
            const blob = new Blob(audioChunks, { type: mimeType });
            if (blob.size < 1000) {
                addMessage("🎤 Too short. Speak for 2+ seconds.", 'bot');
                cleanupStream();
                return;
            }
            showFormatChoice('voice', blob);
            cleanupStream();
        };

        mediaRecorder.onerror = () => {
            addMessage("⚠️ Recording error.", 'bot');
            resetMicUI();
            cleanupStream();
        };

        mediaRecorder.start(1000);
        isRecording = true;
        updateMicUI(true);

        setTimeout(() => {
            if (isRecording && mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
        }, 30000);

    } catch (err) {
        if (err.name === 'NotAllowedError') alert('🎤 Microphone blocked! Allow in browser settings.');
        else if (err.name === 'NotFoundError') alert('🎤 No microphone found!');
        else alert('🎤 Error: ' + err.message);
        resetMicUI();
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    isRecording = false;
    updateMicUI(false);
}

function cleanupStream() {
    if (recordingStream) {
        recordingStream.getTracks().forEach(t => t.stop());
        recordingStream = null;
    }
}

function updateMicUI(recording) {
    if (!micBtn) return;
    const icon = micBtn.querySelector('i');
    if (recording) {
        micBtn.classList.add('recording');
        if (icon) icon.className = 'fas fa-stop';
        micBtn.title = 'Click to Stop';
    } else {
        micBtn.classList.remove('recording');
        if (icon) icon.className = 'fas fa-microphone';
        micBtn.title = 'Click to Speak';
    }
}

function resetMicUI() {
    isRecording = false;
    updateMicUI(false);
    cleanupStream();
}

if (micBtn) {
    micBtn.addEventListener('click', () => {
        if (!isRecording) startRecording(); else stopRecording();
    });
}

// ══════════════════════════════════════
// 🎯 SEND VOICE TO BACKEND (UPDATED with links & media)
// ══════════════════════════════════════

async function sendVoiceToServer(audioBlob, responseFormat = 'both') {
    showTyping();
    setInputState(false);
    if (micBtn) micBtn.disabled = true;

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('session_id', SESSION_ID);
    formData.append('response_format', responseFormat);

    try {
        const res = await fetch(VOICE_API_URL, { method: 'POST', body: formData });
        const data = await res.json();
        hideTyping();

        if (data.error && !data.reply) {
            addMessage("⚠️ " + (data.error || "Failed."), 'bot');
            return;
        }

        const links = data.links || [];
        const media = data.media || null;

        // Update user message with transcribed text
        if (data.user_text) {
            const msgs = chatMessages.querySelectorAll('.user-message');
            const last = msgs[msgs.length - 1];
            if (last) {
                const bubble = last.querySelector('.msg-bubble p');
                if (bubble && bubble.textContent.includes('Voice sent')) {
                    bubble.textContent = '🎤 "' + data.user_text + '"';
                }
            }
        }

        // Show text reply with links & media
        if (responseFormat === 'text' || responseFormat === 'both') {
            if (data.reply) addMessage(data.reply, 'bot', { links, media });
        }

        if (responseFormat === 'speech') {
            addMessage("🔊 Playing audio response...", 'bot', { links, media });
        }

        if (data.audio_url && (responseFormat === 'speech' || responseFormat === 'both')) {
            playAudioResponse(data.audio_url);
        }

        if (responseFormat === 'speech' && !data.audio_url && data.reply) {
            addMessage("⚠️ Audio unavailable. Here's the text:", 'bot');
            addMessage(data.reply, 'bot', { links, media });
        }

    } catch (err) {
        hideTyping();
        console.error('Voice error:', err);
        addMessage("⚠️ Voice failed. Try typing instead.", 'bot');
    } finally {
        setInputState(true);
        if (micBtn) micBtn.disabled = false;
        userInput.focus();
    }
}

// ══════════════════════════════════════
// 🚀 EVENT LISTENERS + INIT
// ══════════════════════════════════════

if (userInput) userInput.addEventListener('input', updateCharCount);

document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === '/') { e.preventDefault(); if (userInput) userInput.focus(); }
    if (e.key === 'Escape') {
        const s = document.getElementById('sidebar');
        if (s && s.classList.contains('open')) toggleSidebar();
        if (isRecording) stopRecording();
        closeImagePreview();
        const choice = document.getElementById('formatChoice');
        if (choice) choice.remove();
    }
});

window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
        const s = document.getElementById('sidebar');
        const o = document.getElementById('sidebarOverlay');
        if (s) s.classList.remove('open');
        if (o) o.classList.remove('show');
    }
});

document.addEventListener('DOMContentLoaded', () => {
    console.log('🎓 JECRC Chatbot + Voice + Links + Media Ready');
    if (userInput) userInput.focus();
    updateCharCount();
    scrollToBottom();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        if (micBtn) { micBtn.style.opacity = '0.4'; micBtn.disabled = true; micBtn.title = 'Mic not supported'; }
    }
});

window.addEventListener('beforeunload', () => {
    if (isRecording) stopRecording();
    cleanupStream();
});

// ══════════════════════════════════════
// 🎨 THEME TOGGLE
// ══════════════════════════════════════

const THEME_KEY = 'jecrc_chatbot_theme';

function getStoredTheme() {
    return localStorage.getItem(THEME_KEY) || 'dark';
}

function setTheme(theme) {
    const html = document.documentElement;
    const themeIcon = document.getElementById('themeIcon');
    const themeLabel = document.getElementById('themeLabel');

    if (theme === 'light') {
        html.setAttribute('data-theme', 'light');
        if (themeIcon) { themeIcon.classList.remove('fa-moon'); themeIcon.classList.add('fa-sun'); }
        if (themeLabel) themeLabel.textContent = 'Light';
        const metaTheme = document.querySelector('meta[name="theme-color"]');
        if (metaTheme) metaTheme.setAttribute('content', '#f0f4f8');
    } else {
        html.removeAttribute('data-theme');
        if (themeIcon) { themeIcon.classList.remove('fa-sun'); themeIcon.classList.add('fa-moon'); }
        if (themeLabel) themeLabel.textContent = 'Dark';
        const metaTheme = document.querySelector('meta[name="theme-color"]');
        if (metaTheme) metaTheme.setAttribute('content', '#0a0e1a');
    }

    localStorage.setItem(THEME_KEY, theme);
}

function toggleTheme() {
    const current = getStoredTheme();
    const newTheme = current === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    const btn = document.getElementById('themeToggleBtn');
    if (btn) {
        btn.style.transform = 'scale(0.9)';
        setTimeout(() => { btn.style.transform = ''; }, 200);
    }
}

(function initTheme() {
    setTheme(getStoredTheme());
})();

// ══════════════════════════════════════
// 🎉 END
// ══════════════════════════════════════