# 🎓 SETU
### AI-Powered College Helpdesk Chatbot for JECRC Foundation

🚀 **Live Demo:** https://setu-2cn0.onrender.com/

💬 **Supports:** English | हिन्दी | Hinglish

---

# 📖 Overview

SETU is an AI-powered college helpdesk chatbot developed for **JECRC Foundation** to simplify access to college-related information for students, parents, and aspirants.

Instead of searching through multiple webpages, visiting different offices, or waiting for admission helplines, users can simply ask questions in natural language and receive instant responses.

The chatbot understands **English, Hindi, and Hinglish**, making it accessible to a wider audience while providing information related to admissions, departments, faculty, academics, hostel, placements, scholarships, campus facilities, and many other college services.

SETU acts as a digital bridge between students and college information by providing quick, accurate, and interactive assistance 24×7.

---

# ❗ Problem Statement

Students often face difficulties while searching for college-related information because:

- Information is scattered across multiple webpages
- Office timings are limited
- Admission helplines remain busy
- Students repeatedly ask similar questions
- Finding department or faculty information takes time
- Hindi-speaking students struggle with English-only information

SETU addresses these challenges by providing a single AI-powered conversational platform that delivers information instantly.

---

# ✨ Features

## 🤖 AI Chatbot

- Instant responses to student queries
- Intelligent intent recognition
- Hybrid NLP-based response generation
- Natural language understanding

---

## 🌍 Multilingual Support

- English
- Hindi
- Hinglish
- Automatic language detection

---

## 🧠 NLP Engine

- TF-IDF Vectorization
- Cosine Similarity Matching
- Keyword Matching
- Automatic Typo Correction
- Vocabulary-based preprocessing

---

## 🎤 Voice Support

- Voice-to-Text using AssemblyAI
- Text-to-Speech using Google TTS
- Microsoft Edge TTS Support

---

## 📚 Knowledge Base

- 203 predefined intents
- 8,883+ training patterns
- English responses
- Hindi responses
- Faculty directory
- Department information
- College information

---

## 👨‍🏫 Faculty Information

- Faculty Search
- Department-wise Faculty Directory
- 265 Faculty Members
- 9 Academic Departments

---

## 📊 Admin Dashboard

- Secure Admin Login
- Chat Analytics
- Daily Statistics
- Weekly Statistics
- Interactive Charts
- Recent Chat Monitoring
- Unresolved Query Tracking
- CSV Export
- PDF Report Export

---

## 🔒 Security Features

- Rate Limiting
- Input Sanitization
- Admin Authentication
- Environment Variable Protection
- Automatic Chat Cleanup

---

## 📱 Responsive Design

- Desktop Support
- Mobile Friendly
- Lightweight Interface

---

# 🏫 Topics Covered

The chatbot provides information about:

- Admission Process
- Eligibility Criteria
- Required Documents
- REAP Counselling
- JEE Admission
- Lateral Entry
- Departments
- Faculty Information
- Fee Structure
- Scholarships
- Refund Policy
- Hostel Facilities
- Hostel Rules
- Mess Facilities
- Placements
- Internship
- Training
- Academics
- Examination
- Attendance
- RTU
- Library
- Sports
- Clubs
- Events
- Wi-Fi
- Medical Facilities
- Transport
- Parking
- Anti Ragging
- Student Support
- Emergency Contacts
- Rankings
- Recognition
- Campus Facilities

and many more...

---

# 📊 Knowledge Base Coverage

| Category | Topics |
|----------|--------|
| Admission | Admission Process, Eligibility, Documents, REAP, JEE, Reservation |
| Departments | CSE, CSE(AI), AI & DS, IT, ECE, EE, ME, CE |
| Fees | Fee Structure, Scholarships, Refund Policy |
| Placements | Companies, Packages, Training, Internship |
| Hostel | Rooms, Rules, Mess, Visitors, Security |
| Academics | Exams, Attendance, Results, Credits, RTU |
| Campus Life | Library, Clubs, Sports, Medical, Wi-Fi |
| Safety | Anti-Ragging, Grievance, Counselling |
| General | Rankings, Recognition, Facilities |

---

# ⚙️ How It Works

When a user asks a question, SETU follows these steps:

1. Cleans and normalizes the user query.
2. Corrects common spelling mistakes.
3. Detects the input language.
4. Performs TF-IDF Vectorization.
5. Calculates Cosine Similarity.
6. Applies Keyword Matching.
7. Selects the best matching intent.
8. Returns the response in the appropriate language.
9. Stores chat history for analytics.

---

# 🛠 Tech Stack

## Backend

- Python
- Flask

## AI / NLP

- Scikit-Learn
- TF-IDF Vectorization
- Cosine Similarity
- NLTK
- RapidFuzz

## Database

- SQLite

## Frontend

- HTML
- CSS
- JavaScript

## Voice AI

- AssemblyAI
- Google TTS (gTTS)
- Microsoft Edge TTS

## Other Libraries

- BeautifulSoup4
- Requests
- FPDF2

## Deployment

- Render
- GitHub

---

# 📂 Project Structure

```text
Setu/
│
├── app.py
├── chatbot_engine.py
├── database.py
├── config.py
├── faculty_db.py
├── faculty_data.json
├── intentsupdated.json
├── web_scraper.py
├── requirements.txt
├── render.yaml
├── Procfile
├── api.env
├── chat_history.db
│
├── templates/
│   ├── index.html
│   ├── admin.html
│   ├── admin_login.html
│   └── chatbot_widget.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── README.md
```

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/heydaksh7297/Setu.git

cd Setu
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create an `api.env` file and add:

```text
ASSEMBLYAI_API_KEY=YOUR_API_KEY
```

---

## Run Application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 🌐 Live Demo

https://setu-2cn0.onrender.com/

---

# 📈 Performance

| Metric | Value |
|---------|-------:|
| Total Intents | 203 |
| Training Patterns | 8,883+ |
| Faculty Members | 265 |
| Departments | 9 |
| Overall Accuracy | 85–90% |
| Exact Match Accuracy | ~95% |
| Hybrid Model Accuracy | ~92% |
| Hindi Detection | >90% |
| Average Response Time | <1 Second |

---

# 📌 Key Highlights

- AI-Powered College Helpdesk
- English, Hindi & Hinglish Support
- Intelligent Intent Recognition
- TF-IDF Based NLP
- Keyword Matching
- Typo Correction
- Voice Chat
- Faculty Search
- SQLite Chat History
- Admin Dashboard
- Live Analytics
- CSV Export
- PDF Export
- Responsive Design
- REST APIs
- Automatic Data Cleanup

---

# 🎯 Applications

- College Helpdesk
- Admission Assistance
- Student Support
- Faculty Information System
- FAQ Automation
- Campus Information Portal

---

# 🔮 Future Scope

- LLM Integration (ChatGPT/Gemini)
- WhatsApp Bot
- ERP Integration
- Android Application
- iOS Application
- Self-Learning AI
- Regional Language Support
- Website Widget
- Sentiment Detection
- Document Processing

---

# 👨‍💻 Developer

**Daksh**

B.Tech Computer Science & Engineering

JECRC Foundation, Jaipur

GitHub: https://github.com/heydaksh7297
---

# 🤝 Contributing

Contributions, suggestions, bug reports, and feature requests are welcome.

Feel free to fork the repository, create a new branch, and submit a pull request.

---

# ⭐ Support

If you found this project useful, please consider giving this repository a **Star ⭐**.

It helps others discover the project and motivates further development.
