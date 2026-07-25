"""
============================================================
  JECRC Foundation - College Helpdesk AI Chatbot
  Main Flask Application

  🔧 UPDATED:
  - Admin authentication (login/logout)
  - Rate limiting
  - Input sanitization
  - Voice chat (AssemblyAI + gTTS/Microsoft TTS)
  - 🔥 Text-to-Speech API route
  - 🔥 Format choice support (text/speech/both)
  - 🌐 Multi-Language Support (Hindi/English)
  - 🌐 Language toggle API
  - 🌐 Auto language detection
  - 📊 Chart Data API Endpoints
  - 📊 Live Stats API
  - 📊 Unresolved Queries API
  - 📊 Recent Chats API
  - 📥 Export: CSV + PDF Chat History (UPDATED)
  - 📥 Export: Analytics PDF Report (UPDATED)


============================================================
"""

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response
from flask_cors import CORS
from functools import wraps
from collections import defaultdict
import uuid
import datetime
import time
import html
import os
import csv
import io
import asyncio
import edge_tts
from gtts import gTTS
import re
import requests
from dotenv import load_dotenv
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load env files
load_dotenv('.env')
load_dotenv('api.env')

from config import Config
from chatbot_engine import ChatbotEngine
from database import ChatDatabase
from web_scraper import WebScraper

# ── Initialize Flask App ──
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# ── Initialize Components ──
print("\n" + "=" * 60)
print("🤖 JECRC Foundation Helpdesk AI Chatbot")
print("🌐 Multi-Language Support: Hindi + English")
print("📊 Admin Dashboard with Charts & Live Stats")
print("📥 Export: CSV + PDF Reports")
print("=" * 60 + "\n")

chatbot = ChatbotEngine(
    intents_file='intentsupdated.json',
    confidence_threshold=Config.CONFIDENCE_THRESHOLD
)

database = ChatDatabase(db_path=Config.DATABASE_PATH)
scraper = WebScraper()

# Verify API keys
_aai_key = os.getenv("ASSEMBLYAI_API_KEY", "")
print(f"🎤 AssemblyAI Key: {'✅ Loaded' if _aai_key else '❌ MISSING!'}")
print(f"\n✅ All components initialized!")
print(f"🌐 Server ready at http://localhost:{Config.PORT}")
print(f"🔐 Admin: http://localhost:{Config.PORT}/admin")
print("=" * 60 + "\n")

database.cleanup_old_chats(days=90)


# ══════════════════════════════════════
# 🔧 Rate Limiting
# ══════════════════════════════════════
request_counts = defaultdict(list)
last_cleanup = time.time()


def check_rate_limit(ip):
    """Check if IP has exceeded rate limit"""
    global last_cleanup
    now = time.time()

    if now - last_cleanup > 300:
        old_ips = [k for k, v in request_counts.items()
                   if not v or now - v[-1] > 120]
        for old_ip in old_ips:
            del request_counts[old_ip]
        last_cleanup = now

    request_counts[ip] = [t for t in request_counts[ip] if now - t < 60]
    if len(request_counts[ip]) >= Config.RATE_LIMIT_PER_MINUTE:
        return False
    request_counts[ip].append(now)
    return True


# ══════════════════════════════════════
# 🔧 Admin Authentication
# ══════════════════════════════════════
def admin_required(f):
    """Protect admin routes with login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            if request.path.startswith('/api/admin'):
                return jsonify({
                    'error': 'Unauthorized',
                    'message': 'Admin login required'
                }), 401
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


# ══════════════════════════════════════
# 🎤 Speech-to-Text (AssemblyAI)
# ══════════════════════════════════════
def speech_to_text(audio_file_path):
    """Convert audio to text using AssemblyAI"""
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("  ❌ ASSEMBLYAI_API_KEY not found!")
        return None

    try:
        file_size = os.path.getsize(audio_file_path)
        print(f"  📁 File size: {file_size / 1024:.1f} KB")

        if file_size < 100:
            print("  ❌ File too small")
            return None

        # Step 1: Upload
        print("  🎤 Uploading to AssemblyAI...")
        with open(audio_file_path, 'rb') as f:
            file_data = f.read()

        upload_response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={
                "authorization": api_key,
                "content-type": "application/octet-stream"
            },
            data=file_data,
            timeout=60
        )
        print(f"  📡 Upload status: {upload_response.status_code}")

        if upload_response.status_code != 200:
            print(f"  ❌ Upload failed: {upload_response.text[:300]}")
            return None

        audio_url = upload_response.json().get('upload_url')
        if not audio_url:
            print("  ❌ No upload_url")
            return None

        print(f"  ✅ Uploaded!")

        # Step 2: Request transcription
        print("  🧠 Requesting transcription...")
        transcript_response = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers={
                "authorization": api_key,
                "content-type": "application/json"
            },
            json={
                "audio_url": audio_url,
                "speech_models": ["universal-2"]
            },
            timeout=30
        )
        print(f"  📡 Transcript status: {transcript_response.status_code}")

        if transcript_response.status_code != 200:
            print(f"  ❌ Transcript failed: {transcript_response.text[:300]}")
            return None

        transcript_id = transcript_response.json().get('id')
        if not transcript_id:
            print("  ❌ No transcript ID")
            return None

        print(f"  📝 Transcript ID: {transcript_id}")

        # Step 3: Poll for result
        print("  ⏳ Waiting for result...")
        for attempt in range(1, 91):
            time.sleep(1)
            poll = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers={"authorization": api_key},
                timeout=15
            )
            result = poll.json()
            status = result.get('status')

            if status == 'completed':
                text = result.get('text', '')
                print(f"  ✅ Result: '{text}'")
                return text if text and text.strip() else None
            elif status == 'error':
                print(f"  ❌ Error: {result.get('error')}")
                return None

            if attempt % 5 == 0:
                print(f"  ⏳ {status}... ({attempt}s)")

        print("  ❌ Timeout!")
        return None

    except Exception as e:
        print(f"  ❌ STT Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ══════════════════════════════════════
# 🔊 Text-to-Speech (Google TTS - FREE)
# ══════════════════════════════════════

def clean_text_for_speech(text):
    """
    Remove emojis and format text properly for TTS
    🔹 → bullet point (spoken naturally)
    ✅ → removed
    All emojis → stripped
    """
    if not text:
        return ""

    clean = text

    # Step 1: Remove markdown bold markers
    clean = clean.replace('**', '').replace('__', '').replace('*', '')

    # Step 1.5: Fix number ranges for natural speech
    clean = re.sub(
        r'(\d[\d,.:]*)\s*[-–—]\s*([\d₹][\d,.:]*)',
        r'\1 to \2',
        clean
    )

    clean = re.sub(
        r'(\d+)\s*[-–—]\s*(\d+)(%)',
        r'\1 to \2\3',
        clean
    )

    # Fix "₹" symbol for speech
    clean = clean.replace('₹', 'rupees ')

    # Fix "LPA" to be spoken clearly
    clean = re.sub(r'\bLPA\b', 'L P A', clean)

    # Fix "+" to "plus"
    clean = re.sub(r'(\d)\+', r'\1 plus', clean)

    # Step 2: Replace bullet-style emojis with "Point:" or just dash
    bullet_emojis = [
        '🔹', '🔸', '▪️', '▫️', '◾', '◽', '◆', '◇',
        '●', '○', '•', '►', '▸', '➤', '➜', '➡️',
        '👉', '📌', '📍', '🔘',
    ]
    for emoji in bullet_emojis:
        clean = clean.replace(emoji, ' - ')

    # Step 3: Replace numbered/label emojis with readable text
    label_replacements = {
        '1️⃣': '1.', '2️⃣': '2.', '3️⃣': '3.', '4️⃣': '4.',
        '5️⃣': '5.', '6️⃣': '6.', '7️⃣': '7.', '8️⃣': '8.',
        '9️⃣': '9.', '🔟': '10.',
        '✅': '', '❌': 'Not allowed,',
        '⚠️': 'Important,', '🚫': 'Not allowed,',
        '💡': 'Tip,', '📞': 'Phone:', '📧': 'Email:',
        '📍': 'Address:', '🌐': 'Website:',
        '📋': '', '📊': '', '📈': '', '📉': '',
        '💰': '', '💳': '', '🏦': '',
        '🎓': '', '🏛️': '', '🏫': '', '🏠': '',
        '💼': '', '🎯': '', '🎉': '', '🏆': '',
        '🤖': '', '👋': '', '😊': '', '😅': '',
        '🤔': '', '🙁': '', '👈': '', '🔥': '',
        '💚': '', '👩': '', '♿': '', '⚖️': '',
        '📜': '', '👔': '', '📱': '', '📧': '',
        '🚨': '', '🏧': '', '📅': '', '🌧️': '',
        '🍕': '', '🎂': '', '💻': '', '📡': '',
        '⚡': '', '⚙️': '', '🏗️': '', '🔒': '',
        '📝': '', '📚': '', '🔬': '', '🕐': '',
        '📶': '', '🚌': '', '🅿️': '', '🎭': '',
        '🌟': '', '🌿': '', '🏅': '', '⭐': '',
        '🇮🇳': '', '🚀': '', '🍽️': '', '🏥': '',
        '🚭': '', '👨‍🏫': '', '👨‍👩‍👧': '', '📄': '',
        '✈️': '', '🆘': '', '💬': '', '🎤': '',
        '🔊': '', '_(': '', ')_': '',
    }
    for emoji, replacement in label_replacements.items():
        clean = clean.replace(emoji, replacement)

    # Step 4: Remove ALL remaining emojis using regex
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U0000FE00-\U0000FE0F"
        "\U0000200D"
        "\U00002000-\U0000200F"
        "\U0000205F-\U00002060"
        "\U00002934-\U00002935"
        "\U000025AA-\U000025AB"
        "\U000025FB-\U000025FE"
        "\U00002B05-\U00002B07"
        "\U00002B1B-\U00002B1C"
        "\U00002B50"
        "\U00002B55"
        "\U0000231A-\U0000231B"
        "\U000023E9-\U000023F3"
        "\U000023F8-\U000023FA"
        "\U00003030"
        "\U000000A9"
        "\U000000AE"
        "\U00002122"
        "]+",
        flags=re.UNICODE
    )
    clean = emoji_pattern.sub(' ', clean)

    # Step 5: Clean up extra spaces and formatting
    clean = re.sub(r'\s*-\s*-\s*', ' - ', clean)
    clean = re.sub(r'\s+', ' ', clean)
    clean = re.sub(r'\n\s*\n', '\n', clean)
    clean = clean.strip()

    return clean

async def generate_edge_tts(text, output_path):
    communicate = edge_tts.Communicate(
        text=text,
        voice="hi-IN-MadhurNeural"
    )
    await communicate.save(output_path)

    
def text_to_speech(text, language='en'):
    try:
        clean_text = clean_text_for_speech(text)

        if not clean_text or len(clean_text) < 2:
            clean_text = "Here is the information you requested."

        if len(clean_text) > 3000:
            clean_text = clean_text[:3000] + "..."

        audio_filename = f"audio_{uuid.uuid4().hex[:12]}.mp3"
        audio_path = os.path.join("static", audio_filename)

        os.makedirs("static", exist_ok=True)

        # Hindi → Edge TTS
        if language == 'hi':
            print(f"🔊 Generating Hindi audio with Edge TTS...")
            asyncio.run(
                generate_edge_tts(
                    clean_text,
                    audio_path
                )
            )

        # English → gTTS
        else:
            print(f"🔊 Generating English audio with gTTS...")

            tts = gTTS(
                text=clean_text,
                lang='en',
                slow=False
            )

            tts.save(audio_path)

        audio_size = os.path.getsize(audio_path)

        print(
            f"✅ Audio ready ({audio_size / 1024:.1f} KB)"
        )

        return f"/static/{audio_filename}"

    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None
# ══════════════════════════════════════
# 🧹 PDF Helper: Clean text for PDF
# ══════════════════════════════════════
def clean_text_for_pdf(text):
    """Remove ALL non-latin characters for PDF compatibility"""
    if not text:
        return ""
    import re
    clean = str(text)

    # Remove newlines (THIS IS THE MAIN FIX)
    clean = clean.replace('\n', ' ').replace('\r', '')

    # Remove markdown
    clean = clean.replace('**', '').replace('__', '').replace('*', '')

    # Remove all emojis and non-ASCII
    # SAFEST method: keep only printable latin characters
    try:
        clean = clean.encode('latin-1', errors='ignore').decode('latin-1')
    except Exception:
        clean = ''.join(c for c in clean if ord(c) < 256)

    # Clean extra spaces
    clean = re.sub(r'\s+', ' ', clean).strip()

    return clean


# ══════════════════════════════════════
# 🎤 Voice Chat Route
# ══════════════════════════════════════
@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    """Voice chat endpoint - with format choice + language + links & media"""
    file_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({
                "error": "No audio file received",
                "reply": "No audio received. Please try again. 🎤",

            }), 400

        audio_file = request.files['audio']
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        response_format = request.form.get('response_format', 'both')
        language = request.form.get('language', 'auto')

        audio_file.seek(0, 2)
        file_size = audio_file.tell()
        audio_file.seek(0)

        if file_size < 500:
            return jsonify({
                "reply": "Recording too short. Speak for at least 2 seconds. 🎤",
                "user_text": "",
                "audio_url": None,
            }), 400

        if file_size > 25 * 1024 * 1024:
            return jsonify({
                "reply": "Recording too long (max 25MB). 🎤",
                "user_text": "",
                "audio_url": None,
            }), 400

        print(f"\n🎤 Voice Request ({file_size / 1024:.1f} KB, lang={language})")

        # Save temp file
        content_type = audio_file.content_type or 'audio/webm'
        ext = '.webm'
        if 'ogg' in content_type:
            ext = '.ogg'
        elif 'mp4' in content_type:
            ext = '.mp4'
        elif 'wav' in content_type:
            ext = '.wav'

        file_path = os.path.join(os.getcwd(), f"temp_{uuid.uuid4().hex[:12]}{ext}")
        audio_file.save(file_path)

        # Step 1: Speech → Text
        user_text = speech_to_text(file_path)

        if not user_text or not user_text.strip():
            return jsonify({
                "user_text": "",
                "reply": "Couldn't hear you. Please speak clearly and try again. 🎤",
                "audio_url": None,

            })

        print(f"  🗣️ '{user_text}'")

        # Step 2: Chatbot Response (with language + links + media)
        result = chatbot.get_response(user_text, user_id=session_id, language=language)
        bot_reply = result.get('reply', 'Sorry, could not process that.')
        response_lang = result.get('language', 'en')
        print(f"  🤖 Intent: {result.get('intent')} ({result.get('confidence', 0):.0%}) | Lang: {response_lang}")

        # Step 3: Generate audio (in correct language)
        audio_url = None
        if response_format in ('speech', 'both'):
            audio_url = text_to_speech(bot_reply, language=response_lang)

        # Step 4: Save to DB
        try:
            database.save_chat(
                session_id=session_id,
                user_message=f"🎤 {user_text}",
                bot_response=bot_reply,
                intent=result.get('intent', 'voice'),
                confidence=result.get('confidence', 0.0),
                method=f"voice_{result.get('method', 'unknown')}",
                ip_address=request.remote_addr or '0.0.0.0',
                user_agent=request.headers.get('User-Agent', '')[:200]
            )
        except Exception:
            pass

        # ✅ NAYA:
        return jsonify({
            "user_text": user_text,
            "reply": bot_reply,
            "audio_url": audio_url,
            "intent": result.get('intent', ''),
            "confidence": result.get('confidence', 0.0),
            "response_format": response_format,
            "language": response_lang,
            "timestamp": datetime.datetime.now().isoformat()
        })
    # ✅ NAYA:
    except Exception as e:
        print(f"  ❌ Voice Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "reply": "Voice processing failed. Try typing instead. 😅",
            "user_text": "",
            "audio_url": None
        }), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass



# ══════════════════════════════════════
# 🔥 Text-to-Speech API Route
# ══════════════════════════════════════
@app.route('/api/text-to-speech', methods=['POST'])
def tts_endpoint():
    """Convert text to speech - supports multi-language"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data"}), 400

        text = data.get('text', '')
        language = data.get('language', 'en')

        if not text or not text.strip():
            return jsonify({"error": "No text provided"}), 400

        print(f"\n🔊 TTS Request ({len(text)} chars, lang={language})")
        audio_url = text_to_speech(text, language=language)

        if audio_url:
            return jsonify({
                "audio_url": audio_url,
                "language": language,
                "status": "success"
            })
        else:
            return jsonify({
                "error": "Audio generation failed",
                "status": "error"
            }), 500

    except Exception as e:
        print(f"  ❌ TTS endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════
# 🌐 Language API Routes
# ══════════════════════════════════════
@app.route('/api/set-language', methods=['POST'])
def set_language():
    """Set language preference for a user session"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data"}), 400

        language = data.get('language', 'en')
        session_id = data.get('session_id', str(uuid.uuid4()))

        if language not in ['en', 'hi', 'auto']:
            return jsonify({
                "error": "Unsupported language. Use 'en', 'hi', or 'auto'."
            }), 400

        if language != 'auto':
            chatbot.set_user_language(session_id, language)

        session['language'] = language

        print(f"🌐 Language set: {session_id} → {language}")

        return jsonify({
            "status": "success",
            "language": language,
            "message": f"Language set to {'Hindi' if language == 'hi' else 'English' if language == 'en' else 'Auto-detect'}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-language', methods=['GET'])
def get_language():
    """Get current language preference"""
    try:
        session_id = request.args.get('session_id', '')
        language = 'en'

        if session_id:
            language = chatbot.get_user_language(session_id)
        elif 'language' in session:
            language = session['language']

        return jsonify({
            "language": language,
            "supported_languages": [
                {"code": "en", "name": "English", "native": "English"},
                {"code": "hi", "name": "Hindi", "native": "हिंदी"},
                {"code": "auto", "name": "Auto-Detect", "native": "Auto"}
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════
# 🧹 Cleanup Old Audio Files
# ══════════════════════════════════════
@app.route('/api/admin/cleanup-audio', methods=['POST'])
@admin_required
def cleanup_audio():
    """Remove old generated audio files from static folder"""
    try:
        count = 0
        now = time.time()
        static_dir = "static"
        for filename in os.listdir(static_dir):
            if filename.startswith("audio_") and filename.endswith(".mp3"):
                filepath = os.path.join(static_dir, filename)
                if now - os.path.getmtime(filepath) > 3600:
                    os.remove(filepath)
                    count += 1
        return jsonify({
            "status": "success",
            "message": f"Cleaned up {count} old audio files"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════
# ROUTES - Main Pages
# ══════════════════════════════════════
@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html', config=Config)


@app.route('/widget')
def widget():
    return render_template('chatbot_widget.html', config=Config)


# ══════════════════════════════════════
# Admin Login/Logout Routes
# ══════════════════════════════════════
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if username == Config.ADMIN_USERNAME and password == Config.ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            session['admin_login_time'] = datetime.datetime.now().isoformat()
            print(f"✅ Admin login: {username} from {request.remote_addr}")
            return redirect(url_for('admin_dashboard'))
        else:
            error = "Invalid username or password!"
            print(f"❌ Failed admin login attempt from {request.remote_addr}")

    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))

    return render_template('admin_login.html', error=error, config=Config)


@app.route('/admin/logout')
def admin_logout():
    username = session.get('admin_username', 'unknown')
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    session.pop('admin_login_time', None)
    print(f"👋 Admin logout: {username}")
    return redirect(url_for('admin_login'))


@app.route('/admin')
@admin_required
def admin_dashboard():
    return render_template('admin.html', config=Config)


# ══════════════════════════════════════
# API ROUTES - Chat (with multi-language)
# ══════════════════════════════════════
# ═══ PURANA HATAO, YE NAYA PASTE KARO ═══

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat API"""
    try:
        ip = request.remote_addr or '0.0.0.0'
        if not check_rate_limit(ip):
            return jsonify({
                'error': 'Rate limit exceeded',
                'reply': 'Thoda slow karo! 😅 Please wait a moment.'
            }), 429

        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'No message provided',
                'reply': 'Please send a message!'
            }), 400

        user_message = html.escape(data['message'].strip())
        session_id = data.get('session_id', str(uuid.uuid4()))
        language = data.get('language', 'auto')

        if len(user_message) > Config.MAX_MESSAGE_LENGTH:
            return jsonify({
                'reply': f'Message too long! Keep it under {Config.MAX_MESSAGE_LENGTH} chars. 📝',
                'intent': 'error',
                'confidence': 0.0,
                'method': 'validation',
                'language': language
            }), 400

        if not user_message:
            return jsonify({
                'reply': 'Please type something! 😊',
                'intent': 'empty',
                'confidence': 0.0,
                'method': 'validation',
                'language': language
            })

        result = chatbot.get_response(
            user_message,
            user_id=session_id,
            language=language
        )

        chat_id = database.save_chat(
            session_id=session_id,
            user_message=user_message,
            bot_response=result.get('reply', ''),
            intent=result.get('intent', 'default'),
            confidence=result.get('confidence', 0.0),
            method=result.get('method', 'unknown'),
            ip_address=ip,
            user_agent=request.headers.get('User-Agent', '')[:200]
        )

        return jsonify({
            'reply': result.get('reply', ''),
            'intent': result.get('intent', 'default'),
            'confidence': result.get('confidence', 0.0),
            'method': result.get('method', 'unknown'),
            'chat_id': chat_id,
            'session_id': session_id,
            'language': result.get('language', 'en'),
            'timestamp': datetime.datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ Chat API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'reply': 'Sorry, something went wrong. Please try again! 🙏'
        }), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        chat_id = data.get('chat_id')
        rating = data.get('rating', 3)
        comment = html.escape(data.get('comment', ''))

        if chat_id:
            database.save_feedback(chat_id, rating, comment)
            return jsonify({'status': 'success', 'message': 'Thank you! 🙏'})
        return jsonify({'status': 'error', 'message': 'Missing chat_id'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ══════════════════════════════════════
# API ROUTES - Admin (PROTECTED)
# ══════════════════════════════════════
@app.route('/api/admin/analytics', methods=['GET'])
@admin_required
def analytics():
    try:
        data = database.get_analytics()
        stats = chatbot.get_stats()
        data['chatbot_stats'] = stats
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/history', methods=['GET'])
@admin_required
def chat_history():
    try:
        limit = request.args.get('limit', Config.MAX_CHAT_HISTORY, type=int)
        session_id = request.args.get('session_id', None)
        history = database.get_chat_history(session_id=session_id, limit=limit)
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/scrape', methods=['POST'])
@admin_required
def trigger_scrape():
    try:
        data = scraper.scrape_all()
        return jsonify({
            'status': 'success',
            'message': 'Website scraped successfully!',
            'sections': len(data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/resolve', methods=['POST'])
@admin_required
def resolve_query():
    try:
        data = request.get_json()
        query_id = data.get('query_id')
        admin_response = html.escape(data.get('response', ''))
        database.resolve_query(query_id, admin_response)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ══════════════════════════════════════
# 📊 Chart Data API Endpoints
# ══════════════════════════════════════

@app.route('/api/admin/chart-data', methods=['GET'])
@admin_required
def chart_data():
    """Get all chart data for admin dashboard"""
    try:
        chart_type = request.args.get('type', 'all')
        days = request.args.get('days', 30, type=int)

        data = {}

        if chart_type in ('all', 'daily'):
            data['daily_chats'] = database.get_daily_chat_counts(days=days)

        if chart_type in ('all', 'hourly'):
            data['hourly_distribution'] = database.get_hourly_distribution()

        if chart_type in ('all', 'methods'):
            data['method_distribution'] = database.get_method_distribution()

        if chart_type in ('all', 'confidence'):
            data['confidence_trends'] = database.get_confidence_trends(days=days)

        if chart_type in ('all', 'intents'):
            data['top_intents'] = database.get_top_intents(limit=10)

        if chart_type in ('all', 'sessions'):
            data['session_stats'] = database.get_session_stats()

        if chart_type in ('all', 'weekly'):
            data['weekly_comparison'] = database.get_weekly_comparison()

        return jsonify(data)

    except Exception as e:
        print(f"❌ Chart data error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/live-stats', methods=['GET'])
@admin_required
def live_stats():
    """Get real-time live statistics"""
    try:
        analytics_data = database.get_analytics()
        session_stats = database.get_session_stats()
        weekly = database.get_weekly_comparison()
        chatbot_stats = chatbot.get_stats()

        return jsonify({
            'total_chats': analytics_data['total_chats'],
            'today_chats': analytics_data['today_chats'],
            'unique_sessions': analytics_data['unique_sessions'],
            'avg_confidence': analytics_data['avg_confidence'],
            'unresolved_count': analytics_data['unresolved_count'],
            'today_sessions': session_stats['today_sessions'],
            'avg_chats_per_session': session_stats['avg_chats_per_session'],
            'peak_hour': session_stats['peak_hour'],
            'resolve_rate': session_stats['resolve_rate'],
            'weekly_change': weekly['change_percent'],
            'this_week': weekly['this_week'],
            'total_intents': chatbot_stats['total_intents'],
            'total_patterns': chatbot_stats['total_patterns'],
            'timestamp': datetime.datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/unresolved', methods=['GET'])
@admin_required
def unresolved_queries():
    """Get unresolved queries for dashboard"""
    try:
        queries = database.get_unresolved_queries_detailed(limit=30)
        return jsonify({'queries': queries})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/recent-chats', methods=['GET'])
@admin_required
def recent_chats():
    """Get recent chats for live view"""
    try:
        limit = request.args.get('limit', 20, type=int)
        chats = database.get_recent_chats(limit=limit)
        return jsonify({'chats': chats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════
# 📥 NEW: Chat History Export (CSV + PDF)
# ══════════════════════════════════════

@app.route('/api/admin/export/csv', methods=['GET'])
@admin_required
def export_csv():
    """Export chat history as CSV file (no timestamp column)"""
    try:
        limit = request.args.get('limit', 1000, type=int)
        date_from = request.args.get('from', '')
        date_to = request.args.get('to', '')
        intent_filter = request.args.get('intent', '')

        chats = database.get_export_data(
            limit=limit,
            date_from=date_from,
            date_to=date_to,
            intent_filter=intent_filter
        )

        output = io.StringIO()
        writer = csv.writer(output)

        # Header row (NO timestamp)
        writer.writerow([
            'ID', 'Session ID', 'User Message', 'Bot Response',
            'Intent', 'Confidence', 'Method'
        ])

        # Data rows (NO timestamp)
        for chat_row in chats:
            writer.writerow([
                chat_row.get('id', ''),
                chat_row.get('session_id', ''),
                chat_row.get('user_message', ''),
                chat_row.get('bot_response', ''),
                chat_row.get('intent', ''),
                round(chat_row.get('confidence', 0), 4),
                chat_row.get('method', '')
            ])

        output.seek(0)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"jecrc_chat_history_{ts}.csv"

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'text/csv; charset=utf-8'
            }
        )

    except Exception as e:
        print(f"❌ CSV Export error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/export/pdf', methods=['GET'])
@admin_required
def export_pdf():
    """Export chat history as PDF - BULLETPROOF version"""
    try:
        limit = request.args.get('limit', 500, type=int)
        date_from = request.args.get('from', '')
        date_to = request.args.get('to', '')
        intent_filter = request.args.get('intent', '')

        chats = database.get_export_data(
            limit=limit,
            date_from=date_from,
            date_to=date_to,
            intent_filter=intent_filter
        )
        analytics = database.get_analytics()
        stats = chatbot.get_stats()

        from fpdf import FPDF

        def safe_text(text, max_len=50):
            """Make text 100% safe for PDF"""
            if not text:
                return "-"
            try:
                t = str(text)
                t = t.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                t = t.replace('**', '').replace('__', '').replace('*', '')
                # Keep ONLY basic ASCII characters (safest)
                result = ''
                for ch in t:
                    code = ord(ch)
                    if 32 <= code <= 126:  # Basic printable ASCII only
                        result += ch
                    elif ch in ['₹']:
                        result += 'Rs.'
                    else:
                        result += ' '
                # Clean spaces
                import re
                result = re.sub(r'\s+', ' ', result).strip()
                if not result:
                    return "-"
                return result[:max_len]
            except Exception:
                return "-"

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # ══════ PAGE 1: SUMMARY ══════
        pdf.add_page()

        # Header
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(59, 130, 246)
        pdf.cell(0, 12, 'JECRC Foundation - Chat Report', ln=True, align='C')

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(100, 100, 100)
        now_str = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
        pdf.cell(0, 7, f'Generated: {now_str}', ln=True, align='C')
        pdf.cell(0, 7, 'Project: JECRC Foundation Helpdesk AI', ln=True, align='C')

        pdf.ln(5)
        pdf.set_draw_color(59, 130, 246)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

        # Summary Stats
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, 'Summary', ln=True)
        pdf.ln(3)

        try:
            total_chats = analytics.get('total_chats', 0) or 0
            today_chats = analytics.get('today_chats', 0) or 0
            unique_sess = analytics.get('unique_sessions', 0) or 0
            avg_conf = analytics.get('avg_confidence', 0) or 0
            unresolved = analytics.get('unresolved_count', 0) or 0
            total_int = stats.get('total_intents', 0) or 0

            summary_data = [
                ('Total Chats', str(total_chats)),
                ("Today's Chats", str(today_chats)),
                ('Unique Sessions', str(unique_sess)),
                ('Average Confidence', f"{float(avg_conf) * 100:.1f}%"),
                ('Unresolved Queries', str(unresolved)),
                ('Total Intents', str(total_int)),
                ('Records in this export', str(len(chats))),
            ]

            for label, value in summary_data:
                pdf.set_font('Helvetica', '', 10)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(80, 7, str(label))
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(30, 30, 30)
                pdf.cell(0, 7, str(value), ln=True)
        except Exception as e:
            print(f"  ⚠️ Summary error: {e}")
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 7, 'Summary data unavailable', ln=True)

        pdf.ln(5)

        # Top Intents
        try:
            top_intents = analytics.get('top_intents', [])
            if top_intents:
                pdf.set_font('Helvetica', 'B', 14)
                pdf.set_text_color(30, 30, 30)
                pdf.cell(0, 10, 'Top 10 Topics', ln=True)
                pdf.ln(2)

                pdf.set_font('Helvetica', 'B', 9)
                pdf.set_fill_color(240, 240, 245)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(15, 8, '#', border=1, fill=True, align='C')
                pdf.cell(100, 8, 'Topic', border=1, fill=True)
                pdf.cell(40, 8, 'Count', border=1, fill=True, align='C')
                pdf.ln()

                pdf.set_font('Helvetica', '', 9)
                pdf.set_text_color(60, 60, 60)
                for idx, intent in enumerate(top_intents[:10], 1):
                    name = safe_text(intent.get('intent', 'unknown'), 50)
                    name = name.replace('_', ' ').title()
                    count = str(intent.get('count', 0) or 0)
                    pdf.cell(15, 7, str(idx), border=1, align='C')
                    pdf.cell(100, 7, name, border=1)
                    pdf.cell(40, 7, count, border=1, align='C')
                    pdf.ln()
        except Exception as e:
            print(f"  ⚠️ Top intents error: {e}")

        # ══════ PAGE 2+: CHAT HISTORY ══════
        pdf.add_page()

        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, f'Chat History ({len(chats)} records)', ln=True)
        pdf.ln(3)

        # Column widths
        cw = [12, 58, 58, 32, 18, 18]
        headers = ['#', 'User Message', 'Bot Response', 'Intent', 'Conf', 'Method']

        def draw_header():
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_fill_color(240, 240, 245)
            pdf.set_text_color(50, 50, 50)
            for i, h in enumerate(headers):
                pdf.cell(cw[i], 8, h, border=1, fill=True, align='C')
            pdf.ln()

        draw_header()

        skipped = 0
        for idx, chat in enumerate(chats, 1):
            try:
                # Check page break
                if pdf.get_y() > 260:
                    pdf.add_page()
                    draw_header()

                pdf.set_font('Helvetica', '', 7)
                pdf.set_text_color(60, 60, 60)

                user_msg = safe_text(chat.get('user_message', ''), 35)
                bot_resp = safe_text(chat.get('bot_response', ''), 35)
                intent = safe_text(chat.get('intent', ''), 18)
                method = safe_text(chat.get('method', ''), 10)

                try:
                    conf_val = float(chat.get('confidence', 0) or 0)
                    conf = f"{conf_val * 100:.0f}%"
                except Exception:
                    conf = "0%"

                pdf.cell(cw[0], 7, str(idx), border=1, align='C')
                pdf.cell(cw[1], 7, user_msg, border=1)
                pdf.cell(cw[2], 7, bot_resp, border=1)
                pdf.cell(cw[3], 7, intent, border=1, align='C')
                pdf.cell(cw[4], 7, conf, border=1, align='C')
                pdf.cell(cw[5], 7, method, border=1, align='C')
                pdf.ln()

            except Exception as row_err:
                skipped += 1
                print(f"  ⚠️ Row {idx} skipped: {row_err}")
                continue

        # Footer
        pdf.ln(8)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(150, 150, 150)
        if skipped > 0:
            pdf.cell(0, 6, f'Note: {skipped} rows skipped due to encoding issues', ln=True, align='C')
        pdf.cell(0, 6, 'JECRC Foundation AI Helpdesk | Confidential', ln=True, align='C')

        # Generate PDF bytes
        pdf_bytes = pdf.output()

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"jecrc_chat_report_{ts}.pdf"

        from flask import Response
        return Response(
            bytes(pdf_bytes),
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        print(f"❌ PDF Export error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/export/analytics-pdf', methods=['GET'])
@admin_required
def export_analytics_pdf():
    """Export analytics summary as PDF report"""
    try:
        analytics_data = database.get_analytics()
        stats = chatbot.get_stats()
        daily = database.get_daily_chat_counts(days=7)
        top_intents = database.get_top_intents(limit=10)
        methods = database.get_method_distribution()
        weekly = database.get_weekly_comparison()

        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # ── Header ──
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(59, 130, 246)
        pdf.cell(0, 14, 'JECRC Foundation', ln=True, align='C')
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 9, 'AI Helpdesk Analytics Report', ln=True, align='C')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(140, 140, 140)
        now_str = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
        pdf.cell(0, 7, f'Generated: {now_str} | JECRC Foundation Helpdesk AI', ln=True, align='C')
        pdf.ln(3)
        pdf.set_draw_color(59, 130, 246)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)

        # ── KPI Section ──
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, 'Key Performance Indicators', ln=True)
        pdf.ln(3)

        kpis = [
            ('Total Chats', f"{analytics_data.get('total_chats', 0):,}"),
            ("Today's Chats", str(analytics_data.get('today_chats', 0))),
            ('Unique Sessions', f"{analytics_data.get('unique_sessions', 0):,}"),
            ('Average Confidence', f"{analytics_data.get('avg_confidence', 0) * 100:.1f}%"),
            ('Unresolved Queries', str(analytics_data.get('unresolved_count', 0))),
            ('Total Intents Trained', str(stats.get('total_intents', 0))),
            ('Total Patterns', str(stats.get('total_patterns', 0))),
            ('Vocabulary Size', f"{stats.get('vocabulary_size', 0):,}"),
            ('Weekly Change', f"{'+' if weekly.get('change_percent', 0) >= 0 else ''}{weekly.get('change_percent', 0)}%"),
            ('This Week Chats', str(weekly.get('this_week', 0))),
            ('Last Week Chats', str(weekly.get('last_week', 0))),
        ]

        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(240, 240, 245)
        pdf.cell(95, 8, 'Metric', border=1, fill=True, align='C')
        pdf.cell(95, 8, 'Value', border=1, fill=True, align='C')
        pdf.ln()

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        for label, value in kpis:
            pdf.cell(95, 7, label, border=1)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(95, 7, value, border=1, align='C')
            pdf.set_font('Helvetica', '', 10)
            pdf.ln()

        pdf.ln(8)

        # ── Daily Chats ──
        if daily:
            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, 'Daily Chat Volume (Last 7 Days)', ln=True)
            pdf.ln(2)

            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_fill_color(240, 240, 245)
            pdf.cell(95, 8, 'Date', border=1, fill=True, align='C')
            pdf.cell(95, 8, 'Chat Count', border=1, fill=True, align='C')
            pdf.ln()

            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(60, 60, 60)
            for d in daily:
                pdf.cell(95, 7, str(d.get('date', '')), border=1, align='C')
                pdf.cell(95, 7, str(d.get('count', 0)), border=1, align='C')
                pdf.ln()

            pdf.ln(8)

        # ── Top Intents ──
        if top_intents:
            if pdf.get_y() > 200:
                pdf.add_page()

            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, 'Top 10 Most Asked Topics', ln=True)
            pdf.ln(2)

            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_fill_color(240, 240, 245)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(15, 8, 'Rank', border=1, fill=True, align='C')
            pdf.cell(80, 8, 'Topic / Intent', border=1, fill=True)
            pdf.cell(35, 8, 'Count', border=1, fill=True, align='C')
            pdf.cell(60, 8, 'Avg Confidence', border=1, fill=True, align='C')
            pdf.ln()

            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(60, 60, 60)
            for idx, t in enumerate(top_intents, 1):
                pdf.cell(15, 7, str(idx), border=1, align='C')
                name = str(t.get('intent', '')).replace('_', ' ').title()
                pdf.cell(80, 7, name[:40], border=1)
                pdf.cell(35, 7, str(t.get('count', 0)), border=1, align='C')
                avg_c = t.get('avg_confidence', 0) or 0
                pdf.cell(60, 7, f"{avg_c * 100:.1f}%", border=1, align='C')
                pdf.ln()

            pdf.ln(8)

        # ── Method Distribution ──
        if methods:
            if pdf.get_y() > 220:
                pdf.add_page()

            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, 'Classification Method Distribution', ln=True)
            pdf.ln(2)

            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_fill_color(240, 240, 245)
            pdf.cell(65, 8, 'Method', border=1, fill=True, align='C')
            pdf.cell(65, 8, 'Count', border=1, fill=True, align='C')
            pdf.cell(60, 8, 'Percentage', border=1, fill=True, align='C')
            pdf.ln()

            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(60, 60, 60)
            total = sum(m.get('count', 0) for m in methods)
            for m in methods:
                count = m.get('count', 0)
                pct = (count / total * 100) if total > 0 else 0
                pdf.cell(65, 7, str(m.get('method', 'unknown')), border=1, align='C')
                pdf.cell(65, 7, str(count), border=1, align='C')
                pdf.cell(60, 7, f"{pct:.1f}%", border=1, align='C')
                pdf.ln()

        # Footer
        pdf.ln(15)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, 'Auto-generated by JECRC Foundation AI Helpdesk', ln=True, align='C')
        pdf.cell(0, 6, 'JECRC Foundation Helpdesk AI | Confidential', ln=True, align='C')

        # Generate
        pdf_bytes = pdf.output()

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"jecrc_analytics_report_{ts}.pdf"

        return Response(
            bytes(pdf_bytes),
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        print(f"❌ Analytics PDF error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════
# API ROUTES - Info
# ══════════════════════════════════════
@app.route('/api/info', methods=['GET'])
def info():
    return jsonify({
        'name': Config.BOT_NAME,
        'college': Config.COLLEGE_NAME,
        'version': '1.2.0',
        'project': 'JECRC Foundation Helpdesk AI',
        'features': {
            'multi_language': True,
            'supported_languages': ['en', 'hi', 'auto'],
            'voice_chat': True,
            'text_to_speech': True,
            'admin_charts': True,
            'live_stats': True,
            'export_csv': True,
            'export_pdf': True,
            'export_analytics_pdf': True
        },
        'stats': chatbot.get_stats()
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'languages': ['en', 'hi']
    })


# ══════════════════════════════════════
# Error Handlers
# ══════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    return render_template('index.html', config=Config)


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ══════════════════════════════════════
# Run Application
# ══════════════════════════════════════
if __name__ == '__main__':
    print(f"\n🚀 Starting JECRC Foundation Helpdesk AI Chatbot...")
    print(f"🌐 Open: http://localhost:{Config.PORT}")
    print(f"🔐 Admin: http://localhost:{Config.PORT}/admin/login")
    print(f"🔧 Chat API: http://localhost:{Config.PORT}/api/chat")
    print(f"🎤 Voice API: http://localhost:{Config.PORT}/api/voice-chat")
    print(f"🔊 TTS API: http://localhost:{Config.PORT}/api/text-to-speech")
    print(f"🌐 Language API: http://localhost:{Config.PORT}/api/set-language")
    print(f"📊 Chart API: http://localhost:{Config.PORT}/api/admin/chart-data")
    print(f"📊 Live Stats: http://localhost:{Config.PORT}/api/admin/live-stats")
    print(f"📊 Unresolved: http://localhost:{Config.PORT}/api/admin/unresolved")
    print(f"📊 Recent Chats: http://localhost:{Config.PORT}/api/admin/recent-chats")
    print(f"📥 Export CSV: http://localhost:{Config.PORT}/api/admin/export/csv")
    print(f"📥 Export PDF: http://localhost:{Config.PORT}/api/admin/export/pdf")
    print(f"📥 Analytics PDF: http://localhost:{Config.PORT}/api/admin/export/analytics-pdf")
    print(f"🌐 Languages: English 🇬🇧 + Hindi 🇮🇳 + Auto-detect 🤖")
    print(f"\nPress Ctrl+C to stop.\n")

# app.py ka last section - ye paste karo:

if __name__ == '__main__':
    print(f"\n🚀 Starting JECRC Foundation Helpdesk AI Chatbot...")
    print(f"🌐 Open: http://127.0.0.1:5000")
    print(f"\nPress Ctrl+C to stop.\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )
