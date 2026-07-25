"""
JECRC Foundation - College Helpdesk AI Chatbot
NLP Engine - The AI Brain
Project: J-TECHTRIX 7.0

Features:
- TF-IDF + Cosine Similarity for Intent Classification
- Keyword Matching Fallback
- Hinglish (Hindi-English) Support
- 🔧 Advanced Typo Tolerance (Fuzzy Matching)
- 🔧 Smart "Please Retype" Detection
- Context-Aware Responses
- Hybrid Classification (TF-IDF + Keyword Together)
- 🌐 Multi-Language Support (Hindi/English)
- 🌐 Auto Language Detection
- 🤖 NEW: Context-Aware Follow-up Questions
"""

import json
import random
from faculty_db import FacultyDB
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

for resource, path in [('punkt', 'tokenizers/punkt'), ('wordnet', 'corpora/wordnet')]:
    try:
        nltk.data.find(path)
    except (LookupError, OSError):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            print(f"⚠️ Could not download NLTK '{resource}'. Trying offline...")
            pass

class ChatbotEngine:
    """
    NLP-based Chatbot Engine for JECRC Foundation Helpdesk
    🌐 Multi-Language Support (Hindi/English)
    🤖 NEW: Context-Aware Follow-up Questions
    """
    # ═══════════════════════════════════════
    # 👨‍🏫 NEW: Dynamic Faculty Search
    # ═══════════════════════════════════════
    def _try_faculty_search(self, user_message, language='en'):
        """
        Try to detect if user is asking about faculty/HOD/leadership
        and dynamically search the faculty database
        """
        msg_lower = user_message.lower().strip()

        # ── Quick Check 1: Does message contain "dr." or "prof." prefix? ──
        has_name_prefix = bool(re.search(r'\b(dr\.?|prof\.?|professor|shri)\s+\w+', msg_lower))

        # ── Quick Check 2: Faculty-related keywords ──
        faculty_triggers = {
            # ═══════════════════════════════════════
            # 👨‍🏫 General Faculty Keywords
            # ═══════════════════════════════════════
            'faculty', 'faculties', 'teacher', 'teachers',
            'professor', 'professors', 'prof', 'prof.',
            'staff', 'teaching staff', 'academic staff',
            'faculty list', 'teacher list', 'professor list',
            'all faculty', 'all teachers', 'all professors',
            'faculty members', 'teaching members',
            'faculty details', 'teacher details',
            'faculty info', 'teacher info',
            'faculty search', 'search faculty',
            'find faculty', 'find teacher',
            'show faculty', 'show teachers',
            'faculty dikhao', 'teachers dikhao',
            'faculty batao', 'teachers batao',
            'faculty kaun', 'faculty kon',
            'total faculty', 'kitne faculty',
            'kitne teacher', 'kitne professor',
            'faculty count', 'teacher count',
            'faculty overview', 'faculty summary',
            'faculty strength', 'teaching strength',
            'department faculty', 'dept faculty',
            'faculty of', 'teachers of', 'faculties of',
            'sir', 'madam', 'mam',
            'padane wale', 'padhane wale', 'padhate hai',
            'sikshak', 'shikshak', 'adhyapak', 'guru', 'gurujan',

            # ═══════════════════════════════════════
            # 👑 HOD Keywords — General
            # ═══════════════════════════════════════
            'hod', 'hods', 'all hod', 'all hods',
            'sab hod', 'sabhi hod', 'saare hod',
            'head of department', 'head of dept',
            'department head', 'dept head',
            'hod list', 'hod details', 'hod info',
            'hod kaun', 'hod kon', 'hod kya',
            'hod kaun hai', 'hod kon hai',
            'hod ka naam', 'hod ki list',
            'vibhag adhyaksh', 'vibhag ka head',

            # ═══════════════════════════════════════
            # 👑 HOD — Department Specific
            # ═══════════════════════════════════════
            # CSE
            'hod of cse', 'cse hod', 'cse ka hod', 'cse ki hod',
            'cse ke hod', 'cse branch hod', 'cse department hod',
            'cse hod kaun hai', 'cse ka hod kaun hai',
            'computer science hod', 'computer science ka hod',
            # CSAI
            'hod of csai', 'csai hod', 'csai ka hod', 'csai ki hod',
            'csai ke hod', 'csai branch hod', 'csai hod kaun hai',
            'cs ai hod', 'cse ai hod',
            # AIDS
            'hod of aids', 'aids hod', 'aids ka hod', 'aids ki hod',
            'aids ke hod', 'aids branch hod', 'aids hod kaun hai',
            'ai ds hod', 'data science hod',
            # IT
            'hod of it', 'it hod', 'it ka hod', 'it ki hod',
            'it ke hod', 'it branch hod', 'it department hod',
            'it hod kaun hai', 'it ka hod kaun hai',
            'information technology hod',
            # EE
            'hod of ee', 'ee hod', 'ee ka hod', 'ee ki hod',
            'ee ke hod', 'ee branch hod', 'ee department hod',
            'ee hod kaun hai', 'ee ka hod kaun hai',
            'electrical hod', 'electrical ka hod',
            'electrical engineering hod',
            # ECE
            'hod of ece', 'ece hod', 'ece ka hod', 'ece ki hod',
            'ece ke hod', 'ece branch hod', 'ece department hod',
            'ece hod kaun hai', 'ece ka hod kaun hai',
            'electronics hod', 'electronics ka hod',
            # CE
            'hod of ce', 'ce hod', 'ce ka hod', 'ce ki hod',
            'ce ke hod', 'ce branch hod', 'ce department hod',
            'ce hod kaun hai', 'ce ka hod kaun hai',
            'civil hod', 'civil ka hod', 'civil engineering hod',
            # ME
            'hod of me', 'me hod', 'me ka hod', 'me ki hod',
            'me ke hod', 'me branch hod', 'me department hod',
            'me hod kaun hai', 'me ka hod kaun hai',
            'mechanical hod', 'mechanical ka hod',
            'mechanical engineering hod',
            # First Year
            'hod of first year', 'first year hod', 'first year ka hod',
            'first year ki hod', 'first year ke hod',
            '1st year hod', 'fy hod',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — CSE
            # ═══════════════════════════════════════
            'cse faculty', 'cse faculties', 'cse teachers',
            'cse faculty list', 'cse teacher list', 'cse staff',
            'cse professors', 'cse ke teacher', 'cse ki faculty',
            'cse ka faculty', 'cse department faculty',
            'cse branch faculty', 'cse branch teachers',
            'computer science faculty', 'computer science teachers',
            'computer faculty', 'computer teachers',
            'cse me kaun kaun hai', 'cse me kaun padhata hai',
            'cse wale teacher', 'cse wale sir',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — CSAI
            # ═══════════════════════════════════════
            'csai faculty', 'csai faculties', 'csai teachers',
            'csai faculty list', 'csai teacher list', 'csai staff',
            'csai professors', 'csai ke teacher', 'csai ki faculty',
            'cs ai faculty', 'cs ai teachers',
            'cse ai faculty', 'cse ai teachers',
            'computer science ai faculty',
            'artificial intelligence cse faculty',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — AIDS
            # ═══════════════════════════════════════
            'aids faculty', 'aids faculties', 'aids teachers',
            'aids faculty list', 'aids teacher list', 'aids staff',
            'aids professors', 'aids ke teacher', 'aids ki faculty',
            'ai ds faculty', 'ai ds teachers',
            'data science faculty', 'data science teachers',
            'artificial intelligence data science faculty',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — IT
            # ═══════════════════════════════════════
            'it faculty', 'it faculties', 'it teachers',
            'it faculty list', 'it teacher list', 'it staff',
            'it professors', 'it ke teacher', 'it ki faculty',
            'it department faculty', 'it branch faculty',
            'information technology faculty',
            'information technology teachers',
            'it me kaun kaun hai', 'it wale teacher',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — EE
            # ═══════════════════════════════════════
            'ee faculty', 'ee faculties', 'ee teachers',
            'ee faculty list', 'ee teacher list', 'ee staff',
            'ee professors', 'ee ke teacher', 'ee ki faculty',
            'ee department faculty', 'ee branch faculty',
            'electrical faculty', 'electrical teachers',
            'electrical engineering faculty',
            'electrical engineering teachers',
            'electrical wale teacher', 'electrical ke sir',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — ECE
            # ═══════════════════════════════════════
            'ece faculty', 'ece faculties', 'ece teachers',
            'ece faculty list', 'ece teacher list', 'ece staff',
            'ece professors', 'ece ke teacher', 'ece ki faculty',
            'ece department faculty', 'ece branch faculty',
            'electronics faculty', 'electronics teachers',
            'electronics communication faculty',
            'electronics and communication faculty',
            'electronics wale teacher', 'electronics ke sir',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — CE
            # ═══════════════════════════════════════
            'ce faculty', 'ce faculties', 'ce teachers',
            'ce faculty list', 'ce teacher list', 'ce staff',
            'ce professors', 'ce ke teacher', 'ce ki faculty',
            'ce department faculty', 'ce branch faculty',
            'civil faculty', 'civil teachers',
            'civil engineering faculty', 'civil engineering teachers',
            'civil wale teacher', 'civil ke sir',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — ME
            # ═══════════════════════════════════════
            'me faculty', 'me faculties', 'me teachers',
            'me faculty list', 'me teacher list', 'me staff',
            'me professors', 'me ke teacher', 'me ki faculty',
            'me department faculty', 'me branch faculty',
            'mechanical faculty', 'mechanical teachers',
            'mechanical engineering faculty',
            'mechanical engineering teachers',
            'mechanical wale teacher', 'mechanical ke sir',

            # ═══════════════════════════════════════
            # 🏛️ Department Faculty — First Year
            # ═══════════════════════════════════════
            'first year faculty', 'first year faculties',
            'first year teachers', 'first year staff',
            'first year faculty list', 'first year teacher list',
            'first year professors', 'first year ke teacher',
            'first year ki faculty', 'first year ka faculty',
            '1st year faculty', '1st year faculties',
            '1st year teachers', '1st year staff',
            '1st year faculty list', '1st year teacher list',
            'fy faculty', 'fy teachers', 'fy staff',
            'fy faculty list', 'fy teacher list',
            'pehla saal', 'pehla year', 'pehle saal',
            'first year', '1st year', 'firstyear', '1styear',
            'first year me kaun kaun hai',
            'first year wale teacher', 'first year ke sir',
            'common faculty', 'common teachers',
            'pratham varsh', 'pratham varsh faculty',

            # ═══════════════════════════════════════
            # 📚 Specialization — Mathematics
            # ═══════════════════════════════════════
            'maths teacher', 'maths teachers', 'maths faculty',
            'maths faculties', 'maths sir', 'maths madam',
            'maths mam', 'maths wale', 'maths wali',
            'maths ke teacher', 'maths ki teacher',
            'maths padhane wale', 'maths sikhane wale',
            'math teacher', 'math teachers', 'math faculty',
            'math faculties', 'math sir', 'math madam',
            'mathematics teacher', 'mathematics teachers',
            'mathematics faculty', 'mathematics faculties',
            'mathematics sir', 'mathematics madam',
            'ganit ke teacher', 'ganit wale sir',
            'who teaches maths', 'who teaches mathematics',
            'maths kaun padhata hai', 'maths kaun padhati hai',
            'maths kon padhata', 'maths kon padhati',
            'mathematics kaun padhata', 'math kaun padhata',

            # ═══════════════════════════════════════
            # 📚 Specialization — Physics
            # ═══════════════════════════════════════
            'physics teacher', 'physics teachers', 'physics faculty',
            'physics faculties', 'physics sir', 'physics madam',
            'physics mam', 'physics wale', 'physics wali',
            'physics ke teacher', 'physics ki teacher',
            'physics padhane wale', 'physics sikhane wale',
            'who teaches physics', 'physics kaun padhata hai',
            'physics kaun padhati hai', 'physics kon padhata',
            'bhautiki ke teacher', 'bhautiki wale sir',

            # ═══════════════════════════════════════
            # 📚 Specialization — Chemistry
            # ═══════════════════════════════════════
            'chemistry teacher', 'chemistry teachers',
            'chemistry faculty', 'chemistry faculties',
            'chemistry sir', 'chemistry madam', 'chemistry mam',
            'chemistry wale', 'chemistry wali',
            'chemistry ke teacher', 'chemistry ki teacher',
            'chemistry padhane wale', 'chemistry sikhane wale',
            'chem teacher', 'chem teachers', 'chem faculty',
            'who teaches chemistry', 'chemistry kaun padhata hai',
            'chemistry kaun padhati hai', 'chemistry kon padhata',
            'rasayan ke teacher', 'rasayan wale sir',

            # ═══════════════════════════════════════
            # 📚 Specialization — English & Humanities
            # ═══════════════════════════════════════
            'english teacher', 'english teachers',
            'english faculty', 'english faculties',
            'english sir', 'english madam', 'english mam',
            'english wale', 'english wali',
            'english ke teacher', 'english ki teacher',
            'english padhane wale',
            'humanities teacher', 'humanities teachers',
            'humanities faculty', 'humanities faculties',
            'who teaches english', 'english kaun padhata hai',
            'english kaun padhati hai', 'english kon padhata',
            'angrezi ke teacher', 'angrezi wale sir',

            # ═══════════════════════════════════════
            # 📚 Specialization — Sports
            # ═══════════════════════════════════════
            'sports teacher', 'sports teachers',
            'sports faculty', 'sports faculties',
            'sports sir', 'sports madam', 'sports coach',
            'sports wale', 'sports ke teacher',
            'khel ke teacher', 'khel wale sir',
            'who teaches sports', 'sports kaun padhata',
            'physical education teacher', 'pt teacher',
            'pt sir', 'physical education faculty',

            # ═══════════════════════════════════════
            # 👔 Leadership Keywords
            # ═══════════════════════════════════════
            'chairman', 'chairperson', 'vice chairman',
            'vice chairperson', 'director', 'principal',
            'dean', 'dean mam', 'dean sir', 'dean madam',
            'dean of first year', 'first year dean',
            'dean first year', 'dean 1st year',
            'fy dean', 'dean maam',
            'pradhanacharya', 'nideshak', 'adhyaksh',
            'sabhaapati', 'kul sachiv',
            'chairman kaun hai', 'principal kaun hai',
            'director kaun hai', 'dean kaun hai',
            'chairman kon hai', 'principal kon hai',
            'director kon hai', 'dean kon hai',
            'chairman sir', 'principal sir',
            'director sir', 'principal mam',
            'op agarwal', 'o.p. agarwal', 'o p agarwal',
            'amit agarwal', 'arpit agarwal',
            'shri op agarwal', 'shri amit agarwal',
            'shri arpit agarwal',

            # ═══════════════════════════════════════
            # 🔍 Name Search Keywords
            # ═══════════════════════════════════════
            'who is', 'kaun hai', 'kon hai',
            'about dr', 'about prof', 'about sir', 'about madam',
            'tell me about', 'batao', 'bataiye',
            'ke baare mein', 'ke bare me', 'ke bare mein',
            'details of', 'info about', 'information about',
            'dr.', 'dr ', 'prof.', 'prof ', 'shri',

            # ═══════════════════════════════════════
            # 🏛️ Department List Keywords
            # ═══════════════════════════════════════
            'department list', 'all departments', 'all department',
            'sab department', 'sabhi department', 'saare department',
            'vibhag', 'vibhag list', 'sabhi vibhag',
            'kitne department', 'kitne vibhag',
            'department kitne hai', 'branches',
            'all branches', 'branch list', 'sabhi branch',
            'kitni branch', 'branch kitni hai',
            'department details', 'department info',

            # ═══════════════════════════════════════
            # 🗣️ Hindi / Hinglish General
            # ═══════════════════════════════════════
            'kaun padhata', 'kaun padhati', 'kaun padhaate',
            'kaun sikhata', 'kaun sikhati',
            'kon padhata', 'kon padhati',
            'teacher kaun hai', 'teacher kon hai',
            'faculty kaun hai', 'faculty kon hai',
            'professor kaun hai', 'professor kon hai',
            'teacher batao', 'faculty batao',
            'teacher dikhao', 'faculty dikhao',
            'teacher ka naam', 'faculty ka naam',
            'teacher ki list', 'faculty ki list',
            'teacher ka list', 'faculty ka list',
            'sabhi teacher', 'sabhi faculty',
            'saare teacher', 'saari faculty',
            'teacher chahiye', 'faculty chahiye',
        }
        has_faculty_trigger = any(trigger in msg_lower for trigger in faculty_triggers)

        # ── Quick Check 3: Any known faculty name in message? ──
        has_name_match = False
        if self.faculty_db and not has_faculty_trigger and not has_name_prefix:
            for indexed_name in self.faculty_db.name_to_faculty.keys():
                if len(indexed_name) >= 3 and indexed_name in msg_lower:
                    has_name_match = True
                    break

        if not has_faculty_trigger and not has_name_prefix and not has_name_match:
            return None

        print(f"  👨‍🏫 Faculty search triggered: prefix={has_name_prefix}, keyword={has_faculty_trigger}, name={has_name_match}")

        # Try dynamic search
        result, search_type = self.faculty_db.parse_and_search(user_message)

        if result:
            print(f"  👨‍🏫 Faculty search HIT: type={search_type}")

            if language == 'hi':
                result += "\n\n_(Faculty data English mein hai, jaldi Hindi mein bhi available hoga!)_"

            return {
                'reply': result,
                'intent': f'faculty_dynamic_{search_type}',
                'confidence': 0.95,
                'method': 'faculty_db_search',
                'language': language,
            }

        return None
        # Try dynamic search
        result, search_type = self.faculty_db.parse_and_search(user_message)

        if result:
            print(f"  👨‍🏫 Faculty search hit: type={search_type}")

            # Add Hindi note if needed
            if language == 'hi':
                result += "\n\n_(Faculty data English mein hai, jaldi Hindi mein bhi available hoga!)_"

            return {
                'reply': result,
                'intent': f'faculty_dynamic_{search_type}',
                'confidence': 0.95,
                'method': 'faculty_db_search',
                'language': language,
            }

        return None
    def __init__(self, intents_file='intentsupdated.json', confidence_threshold=0.35):
        """Initialize the chatbot engine"""
        self.lemmatizer = WordNetLemmatizer()
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = 0.80

        self.intents = None
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=8000,
            sublinear_tf=True
        )
        self.intent_vectors = None
        self.pattern_intent_map = []
        self.all_patterns = []
        self.context = {}
        self.conversation_history = {}

        # ═══════════════════════════════════════
        # 🌐 Language Settings
        # ═══════════════════════════════════════
        self.user_language = {}  # user_id → 'en' or 'hi'
        self.supported_languages = ['en', 'hi']
        self.default_language = 'en'




        # Hindi detection keywords
        self.hindi_indicators = {
            'kya', 'hai', 'kaise', 'kab', 'kahan', 'kaun', 'kitna', 'kitne',
            'kitni', 'kyun', 'konsa', 'mujhe', 'mera', 'meri', 'hum', 'aap',
            'batao', 'bataiye', 'bataye', 'chahiye', 'hoga', 'hogi', 'tha',
            'thi', 'the', 'karo', 'karna', 'milega', 'milegi', 'milta',
            'padhai', 'paisa', 'naukri', 'nokri', 'hostal', 'khana',
            'dawakhana', 'chutti', 'achha', 'theek', 'bahut', 'abhi',
            'pehle', 'baad', 'sab', 'koi', 'aur', 'bhi', 'se', 'ka',
            'ki', 'ke', 'ko', 'ne', 'par', 'pe', 'mein', 'yeh', 'woh',
            'toh', 'nahi', 'nhi', 'haan', 'ji', 'matlab', 'yaani',
            'lekin', 'agar', 'toh', 'phir', 'abhi', 'sala', 'yaar',
            'bhai', 'dost', 'accha', 'sahi', 'galat', 'pata', 'samajh',
            'samjha', 'samjhao', 'dekho', 'suno', 'bolo', 'jao', 'aao',
            'lo', 'do', 'lao', 'dikhao', 'padhna', 'likhna', 'sunna',
            'jaana', 'aana', 'dena', 'lena', 'khelna', 'sochna',
            'fees', 'admission', 'hostel', 'placement',
            'kaisa', 'kaisi', 'kahan', 'kidhar', 'idhar', 'udhar',
            'wahan', 'yahan', 'tab', 'jab', 'kal', 'aaj', 'parso',
            'subah', 'shaam', 'raat', 'din', 'mahina', 'saal',
        }

        # ═══════════════════════════════════════
        # 🔧 Typo Corrections Database
        # ═══════════════════════════════════════
        self.typo_corrections = {
            # Admission related typos
            'addmission': 'admission', 'admision': 'admission',
            'admisson': 'admission', 'admissin': 'admission',
            'admisin': 'admission', 'addmision': 'admission',
            'admissoin': 'admission', 'admisssion': 'admission',
            'admishn': 'admission', 'edmission': 'admission',
            # Fee related typos
            'fess': 'fees', 'feee': 'fees', 'feees': 'fees',
            'fes': 'fees', 'feis': 'fees', 'fese': 'fees',
            'pees': 'fees', 'phees': 'fees', 'phes': 'fees',
            'fis': 'fees', 'fii': 'fee', 'fie': 'fee',
            # Placement related typos
            'placment': 'placement', 'plcment': 'placement',
            'plcement': 'placement', 'placemnt': 'placement',
            'plecement': 'placement', 'plasment': 'placement',
            'placemnet': 'placement', 'placemet': 'placement',
            'plasmnt': 'placement',
            # Scholarship typos
            'scholarshp': 'scholarship', 'scholorship': 'scholarship',
            'scholarhip': 'scholarship', 'scholarshup': 'scholarship',
            'scolorship': 'scholarship', 'scholarsip': 'scholarship',
            'sclrship': 'scholarship', 'scholarshiop': 'scholarship',
            # Hostel typos
            'hostle': 'hostel', 'hostl': 'hostel', 'hostol': 'hostel',
            'hostell': 'hostel', 'hosel': 'hostel', 'hstel': 'hostel',
            'hostal': 'hostel', 'hoselt': 'hostel', 'hosteel': 'hostel',
            # Attendance typos
            'attendence': 'attendance', 'attendace': 'attendance',
            'attandance': 'attendance', 'attendense': 'attendance',
            'atendance': 'attendance', 'attendnce': 'attendance',
            'attendanc': 'attendance', 'attendens': 'attendance',
            # Department typos
            'deparment': 'department', 'departmnt': 'department',
            'depatment': 'department', 'departmet': 'department',
            'deprtment': 'department', 'departemnt': 'department',
            # Syllabus typos
            'syllabu': 'syllabus', 'syllabs': 'syllabus',
            'sylabus': 'syllabus', 'syllebus': 'syllabus',
            'syllbus': 'syllabus', 'sillabus': 'syllabus',
            'silabus': 'syllabus', 'syllabas': 'syllabus',
            # Timetable typos
            'timetabel': 'timetable', 'timtable': 'timetable',
            'timatable': 'timetable', 'timetble': 'timetable',
            'timetabl': 'timetable', 'timtabel': 'timetable',
            # Canteen typos
            'cantin': 'canteen', 'canten': 'canteen',
            'cantten': 'canteen', 'cafetaria': 'cafeteria',
            'cafetria': 'cafeteria', 'cafeterea': 'cafeteria',
            'cafetiria': 'cafeteria',
            # Library typos
            'librery': 'library', 'laibrary': 'library',
            'libary': 'library', 'libarary': 'library',
            'liberary': 'library', 'librry': 'library',
            'libray': 'library', 'liberry': 'library',
            # Exam typos
            'examm': 'exam', 'exaam': 'exam', 'exma': 'exam',
            'exams': 'exam', 'exm': 'exam', 'examn': 'exam',
            # Ragging typos
            'raging': 'ragging', 'ragin': 'ragging',
            'raggin': 'ragging', 'rgging': 'ragging',
            # Complaint typos
            'complain': 'complaint', 'complant': 'complaint',
            'complait': 'complaint', 'compalint': 'complaint',
            'compaint': 'complaint', 'cmplaint': 'complaint',
            # Internship typos
            'intership': 'internship', 'internshp': 'internship',
            'intrnship': 'internship', 'internshiip': 'internship',
            'internsip': 'internship', 'intenrship': 'internship',
            # Result typos
            'reult': 'result', 'reslt': 'result', 'rsult': 'result',
            'rezult': 'result', 'reasult': 'result', 'resut': 'result',
            # Transport typos
            'tranport': 'transport', 'transort': 'transport',
            'transprt': 'transport', 'trnsport': 'transport',
            # Accreditation typos
            'accredation': 'accreditation', 'acreditation': 'accreditation',
            'acreditaion': 'accreditation', 'accredtation': 'accreditation',
            # Certificate typos
            'certificte': 'certificate', 'certifcate': 'certificate',
            'cirtificate': 'certificate', 'sertificate': 'certificate',
            # Eligibility typos
            'elegibility': 'eligibility', 'eligiblity': 'eligibility',
            'eligibilty': 'eligibility', 'elgibility': 'eligibility',
            'eligibity': 'eligibility', 'eligblity': 'eligibility',
            # Course typos
            'corse': 'course', 'coures': 'course', 'cource': 'course',
            'coarse': 'course', 'courss': 'course',
            # Branch typos
            'brach': 'branch', 'barnch': 'branch', 'brnch': 'branch',
            'braanch': 'branch',
            # College typos
            'colege': 'college', 'collage': 'college', 'colleg': 'college',
            'collge': 'college', 'colleeg': 'college',
            # Engineering typos
            'enginnering': 'engineering', 'engneering': 'engineering',
            'enginering': 'engineering', 'enginring': 'engineering',
            'engineerng': 'engineering', 'engg': 'engineering',
            # Computer typos
            'compter': 'computer', 'computr': 'computer',
            'comuter': 'computer', 'compueter': 'computer',
            'compuutr': 'computer',
            # Science typos
            'sciene': 'science', 'scince': 'science',
            'scienc': 'science', 'sicence': 'science',
            # Mechanical typos
            'mechancal': 'mechanical', 'mechnaical': 'mechanical',
            'mechancial': 'mechanical', 'mechanicl': 'mechanical',
            # Electrical typos
            'electricl': 'electrical', 'electrcal': 'electrical',
            'electrial': 'electrical', 'electirical': 'electrical',
            # Electronics typos
            'elctronics': 'electronics', 'electroncs': 'electronics',
            'elecrtonics': 'electronics', 'electonics': 'electronics',
            # Sports typos
            'sprots': 'sports', 'spors': 'sports',
            'spoerts': 'sports', 'spports': 'sports',
            # Parking typos
            'parkin': 'parking', 'pakring': 'parking', 'prking': 'parking',
            # Medical typos
            'medcal': 'medical', 'medicl': 'medical', 'mdical': 'medical',
            # WiFi typos
            'wfi': 'wifi', 'wify': 'wifi', 'wie-fi': 'wifi', 'wifii': 'wifi',
            # Alumni typos
            'alumini': 'alumni', 'almuni': 'alumni',
            'alumai': 'alumni', 'alumi': 'alumni',
            # JECRC typos
            'jercr': 'jecrc', 'jerc': 'jecrc', 'jercc': 'jecrc',
            'jcrc': 'jecrc', 'jecr': 'jecrc', 'jecrcc': 'jecrc',
            # Counselling typos
            'counsling': 'counselling', 'counseling': 'counselling',
            'cunselling': 'counselling', 'counselin': 'counselling',
            # Backlog typos
            'baklog': 'backlog', 'backlo': 'backlog',
            'baclog': 'backlog', 'bcklog': 'backlog', 'baklogg': 'backlog',
            # Grievance typos
            'greivance': 'grievance', 'greviance': 'grievance',
            'greivence': 'grievance', 'grevance': 'grievance',
            # Convocation typos
            'convacation': 'convocation', 'convocaton': 'convocation',
            'convocaion': 'convocation', 'convokation': 'convocation',
            # Common Hindi-English misspellings
            'kaise': 'how', 'kese': 'how',
            'kitna': 'how much', 'kitne': 'how many',
            'kitni': 'how much', 'chahiye': 'need',
            'chaiye': 'need', 'chahie': 'need',
            'batao': 'tell', 'btao': 'tell', 'batayo': 'tell',
            'milega': 'available', 'millega': 'available',
            'milga': 'available',
        }

        # Hinglish word mappings
        self.hinglish_map = {
            'kaise': 'how', 'kya': 'what', 'kab': 'when',
            'kahan': 'where', 'kaun': 'who', 'kitna': 'how much',
            'kitne': 'how many', 'kitni': 'how much', 'kyun': 'why',
            'konsa': 'which', 'chahiye': 'need want', 'hai': 'is',
            'hain': 'are', 'tha': 'was', 'hoga': 'will be',
            'batao': 'tell', 'bataiye': 'tell', 'bataye': 'tell',
            'milega': 'available get', 'milegi': 'available get',
            'milta': 'available', 'karna': 'do', 'lena': 'take',
            'dena': 'give', 'janna': 'know',
            'padhai': 'study academics', 'paisa': 'fee money',
            'naukri': 'job placement', 'nokri': 'job placement',
            'parhai': 'study', 'hostal': 'hostel',
            'khana': 'food mess', 'bus': 'transport bus',
            'gaadi': 'vehicle transport', 'dawakhana': 'medical hospital',
            'doctor': 'medical doctor', 'chutti': 'holiday leave',
            'safai': 'cleanliness', 'mujhe': 'i need me',
            'mera': 'my', 'meri': 'my', 'hum': 'we',
            'aap': 'you', 'unka': 'their', 'achha': 'good',
            'theek': 'okay fine', 'bahut': 'very much',
            'abhi': 'now', 'pehle': 'first before',
            'baad': 'after later', 'sab': 'all',
            'koi': 'any someone',
        }

        # ═══════════════════════════════════════
        # 🔧 Devanagari → English Keyword Map
        # ═══════════════════════════════════════
        self.devanagari_keyword_map = {
            'क्लब': 'club clubs', 'क्लब्स': 'clubs', 'क्लबों': 'clubs',
            'एडमिशन': 'admission', 'दाखिला': 'admission',
            'फीस': 'fees fee', 'शुल्क': 'fees fee',
            'प्लेसमेंट': 'placement', 'नौकरी': 'job placement',
            'हॉस्टल': 'hostel', 'छात्रावास': 'hostel',
            'कॉलेज': 'college jecrc', 'विश्वविद्यालय': 'university',
            'कोर्स': 'course', 'पाठ्यक्रम': 'course',
            'ब्रांच': 'branch department', 'शाखा': 'branch',
            'डिपार्टमेंट': 'department', 'विभाग': 'department',
            'परीक्षा': 'exam examination', 'एग्जाम': 'exam',
            'सिलेबस': 'syllabus', 'लाइब्रेरी': 'library',
            'पुस्तकालय': 'library', 'कैंपस': 'campus', 'परिसर': 'campus',
            'स्पोर्ट्स': 'sports', 'खेल': 'sports games',
            'बस': 'bus transport', 'ट्रांसपोर्ट': 'transport',
            'पार्किंग': 'parking', 'वाईफाई': 'wifi',
            'इंटरनेट': 'internet wifi', 'मेडिकल': 'medical',
            'डॉक्टर': 'doctor medical', 'अस्पताल': 'hospital medical',
            'कैंटीन': 'canteen food', 'मेस': 'mess food',
            'खाना': 'food mess canteen', 'भोजन': 'food mess',
            'ऑडिटोरियम': 'auditorium', 'सभागार': 'auditorium',
            'लैब': 'lab laboratory', 'प्रयोगशाला': 'lab laboratory',
            'जिम': 'gym sports', 'अटेंडेंस': 'attendance',
            'उपस्थिति': 'attendance', 'रिजल्ट': 'result',
            'परिणाम': 'result', 'बैकलॉग': 'backlog',
            'सप्लीमेंट्री': 'supplementary backlog',
            'टाइमटेबल': 'timetable schedule',
            'समय': 'timing timetable', 'सारणी': 'timetable',
            'सेमेस्टर': 'semester', 'प्रोजेक्ट': 'project',
            'परियोजना': 'project', 'असाइनमेंट': 'assignment',
            'स्कॉलरशिप': 'scholarship', 'छात्रवृत्ति': 'scholarship',
            'लोन': 'loan education', 'ऋण': 'loan',
            'रिफंड': 'refund', 'पैकेज': 'package placement salary',
            'सैलरी': 'salary package',
            'प्रिंसिपल': 'principal director',
            'डायरेक्टर': 'director principal',
            'चेयरमैन': 'chairman management',
            'फैकल्टी': 'faculty teacher professor',
            'प्रोफेसर': 'professor faculty teacher',
            'टीचर': 'teacher faculty', 'शिक्षक': 'teacher faculty',
            'एलुमनाई': 'alumni', 'पूर्व': 'alumni former',
            'इवेंट': 'event fest', 'कार्यक्रम': 'event',
            'फेस्ट': 'fest event', 'उत्सव': 'fest event',
            'सोसाइटी': 'society clubs', 'रोबोटिक्स': 'robotics club',
            'कोडिंग': 'coding club programming',
            'हैकाथॉन': 'hackathon coding',
            'स्टार्टअप': 'startup entrepreneurship',
            'इंटर्नशिप': 'internship training',
            'ट्रेनिंग': 'training placement',
            'सेमिनार': 'seminar', 'वर्कशॉप': 'workshop',
            'रैगिंग': 'ragging', 'शिकायत': 'complaint grievance',
            'समस्या': 'problem complaint',
            'नियम': 'rules discipline',
            'ड्रेस': 'dress code uniform',
            'यूनिफॉर्म': 'uniform dress code',
            'सुरक्षा': 'safety security',
            'आपातकालीन': 'emergency',
            'सर्टिफिकेट': 'certificate',
            'प्रमाणपत्र': 'certificate',
            'कन्वोकेशन': 'convocation degree',
            'डिग्री': 'degree convocation',
            'दीक्षांत': 'convocation',
            'रिसर्च': 'research', 'अनुसंधान': 'research',
            'पेटेंट': 'patent research',
            'संपर्क': 'contact', 'फोन': 'phone contact',
            'ईमेल': 'email contact', 'पता': 'address location',
            'वेबसाइट': 'website',
            'कैसे': 'how', 'कैसा': 'how', 'कैसी': 'how',
            'कितना': 'how much', 'कितनी': 'how much',
            'कितने': 'how many', 'कब': 'when',
            'कहाँ': 'where', 'कहां': 'where',
            'क्या': 'what', 'कौन': 'who', 'कौनसा': 'which',
            'बताओ': 'tell about', 'बताइए': 'tell about',
            'बताएं': 'tell about', 'बतायें': 'tell about',
            'जानकारी': 'information details',
            'सुविधा': 'facility', 'सुविधाएं': 'facilities',
            'सुविधाएँ': 'facilities',
            'प्रक्रिया': 'process', 'तरीका': 'process method',
            'पात्रता': 'eligibility',
            'दस्तावेज': 'documents', 'दस्तावेज़': 'documents',
            'फॉर्म': 'form application',
            'आवेदन': 'apply application',
            'मदद': 'help', 'सहायता': 'help',
            'में': 'in', 'है': 'is', 'हैं': 'are',
            'के': 'of', 'का': 'of', 'की': 'of',
            'और': 'and', 'या': 'or',
            'अच्छा': 'good', 'बेहतर': 'better',
            'उपलब्ध': 'available', 'मिलेगा': 'available get',
            'कौनसी': 'which', 'कौनसे': 'which',
            'छुट्टी': 'holiday vacation leave',
            'छुट्टियां': 'holidays vacation',
        }

        # Load intents
        self._load_intents(intents_file)
        self._build_vocabulary()
        self._train()
                # ═══════════════════════════════════════
        # 👨‍🏫 NEW: Dynamic Faculty Search System
        # ═══════════════════════════════════════
        try:
            self.faculty_db = FacultyDB('faculty_data.json')
            self.faculty_search_enabled = True
        except Exception as e:
            print(f"⚠️ Faculty DB failed to load: {e}")
            self.faculty_db = None
            self.faculty_search_enabled = False

        print(f"🤖 {self.__class__.__name__} initialized successfully!")
        print(f"  📝 Typo corrections loaded: {len(self.typo_corrections)}")
        print(f"  📝 Vocabulary words: {len(self.vocabulary)}")
        print(f"  🌐 Languages supported: {self.supported_languages}")




    # ═══════════════════════════════════════
    # 🔀 NEW: Multi-Intent Detection
    # ═══════════════════════════════════════
    def _detect_multi_intent(self, user_message, language='en'):
        """Detect if user is asking about multiple topics in one message"""
        msg_lower = user_message.lower().strip()

        # Fix typos first
        corrected_msg, _, _, _ = self._fix_typos(msg_lower)

        # Connectors that indicate multi-topic query
        connectors = {
            'aur', 'and', 'both', 'dono', 'bhi', 'also',
            'plus', 'with', 'ya', 'or', 'ke saath', 'saath',
            'sath', 'along with', 'as well as', 'ke alawa',
            'besides', 'tatha', 'evam', 'ke baare',
        }

        # Check if any connector present
        has_connector = any(f' {c} ' in f' {msg_lower} ' for c in connectors)
        has_comma = ',' in msg_lower

        if not has_connector and not has_comma:
            return None

        # Topic keyword → intent mapping
        topic_intents = {
            'admission': 'admission_process',
            'eligibility': 'admission_eligibility',
            'fees': 'fee_structure',
            'fee': 'fee_structure',
            'placement': 'placement',
            'hostel': 'hostel',
            'mess': 'mess_food',
            'food': 'mess_food',
            'canteen': 'mess_food',
            'library': 'library',
            'sports': 'sports',
            'gym': 'sports',
            'exam': 'exam_schedule',
            'syllabus': 'syllabus',
            'timetable': 'timetable',
            'attendance': 'attendance',
            'result': 'result',
            'scholarship': 'scholarship',
            'loan': 'education_loan',
            'transport': 'transport',
            'bus': 'transport',
            'wifi': 'wifi',
            'internet': 'wifi',
            'parking': 'parking',
            'medical': 'medical',
            'doctor': 'medical',
            'ragging': 'ragging',
            'club': 'clubs',
            'clubs': 'clubs',
            'event': 'events',
            'fest': 'events',
            'department': 'departments',
            'branch': 'departments',
            'course': 'departments',
            'backlog': 'backlog_kt',
            'contact': 'contact',
            'alumni': 'alumni',
            'faculty': 'hod_faculty',
            'teacher': 'hod_faculty',
            'internship': 'internship',
            'cse': 'cse_department',
            'ece': 'ece_department',
            'mechanical': 'me_department',
            'electrical': 'ee_department',
            'civil': 'ce_department',
            'mba': 'mba_program',
            'mca': 'mca_program',
            'campus': 'campus_life',
            'refund': 'fee_refund',
            'document': 'admission_documents',
            'accreditation': 'accreditation',
            'naac': 'accreditation',
        }

        # Find all matching topics in BOTH original and corrected msg
        matched_intents = []
        matched_keywords = []
        for check_msg in [msg_lower, corrected_msg]:
            for keyword, intent_tag in topic_intents.items():
                if keyword in check_msg.split() or keyword in check_msg:
                    if intent_tag not in matched_intents:
                        matched_intents.append(intent_tag)
                        matched_keywords.append(keyword)

        if len(matched_intents) >= 2:
            print(f"  🔀 Multi-intent detected: {list(zip(matched_keywords, matched_intents))}")
            return matched_intents

        return None

    # ═══════════════════════════════════════
    # 🌐 Language Detection
    # ═══════════════════════════════════════
    def detect_language(self, text):
        """Auto-detect if the message is in Hindi/Hinglish or English"""
        if not text:
            return 'en'

        text_lower = text.lower().strip()
        words = text_lower.split()

        if not words:
            return 'en'

        # ──────────────────────────────────────
        # STEP 1: If Devanagari script present → Hindi
        # ──────────────────────────────────────
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        if devanagari_count > len(text) * 0.3:
            return 'hi'

        # ──────────────────────────────────────
        # STEP 2: If NO Devanagari at all → Quick English check
        # If no Hindi script AND no Hinglish words → FORCE English
        # ──────────────────────────────────────
        if devanagari_count == 0:
            common_hinglish = re.search(
                r'\b(kya|kaise|kab|kahan|kitna|kitne|kitni|'
                r'hai|hain|tha|thi|hoga|hogi|'
                r'chahiye|batao|bataiye|bataye|milega|milegi|'
                r'mujhe|mera|meri|humko|humara|humari|'
                r'kaisa|kaisi|konsa|konsi|'
                r'nahi|nhi|aur|lekin|abhi|agar|ya|matlab|'
                r'samajh|pata|accha|bahut|bohot|sab|kuch|'
                r'yahan|wahan|idhar|udhar|'
                r'mein|ke|ka|ki|ko|se|par|pe|bhi|toh|na|'
                r'ho|karke|karenge|karega|karegi|'
                r'bolna|bolte|bolo|pehle|baad|'
                r'dekho|dekh|suno|chalo|aao|jao|raho|karo|'
                r'padh|seekh|likh|padho|seekho|likho|'
                r'haan|ji|theek|sahi|galat|'
                r'paisa|paise|rupee|rupaye|bharni|bharna|'
                r'kharcha|lagegi|lagega|lagti|lagta|'
                r'padegi|padega|milti|milta|hoti|hota|'
                r'wala|wale|wali|walon|'
                r'koi|kaun|kis|kisko|kiska|kiski|'
                r'yeh|ye|wo|woh|isko|usko|isme|usme)\b',
                text_lower
            )
            if not common_hinglish:
                return 'en'

        # ──────────────────────────────────────
        # STEP 3: Word-by-word Hindi vs English count
        # ──────────────────────────────────────
        hindi_word_count = 0
        english_word_count = 0
        total_meaningful_words = 0

        for word in words:
            clean_word = word.strip(string.punctuation).lower()
            if len(clean_word) <= 1:
                continue
            total_meaningful_words += 1
            if clean_word in self.hindi_indicators:
                hindi_word_count += 1
            elif clean_word in self.hinglish_map:
                hindi_word_count += 1
            else:
                english_word_count += 1

        if total_meaningful_words == 0:
            return 'en'

        hindi_ratio = hindi_word_count / total_meaningful_words
        if hindi_ratio >= 0.30:
            return 'hi'

        # ──────────────────────────────────────
        # STEP 4: Hinglish pattern matching
        # ──────────────────────────────────────
        hinglish_patterns = [
            r'\b(kya|kaise|kab|kahan|kitna|kitne|kitni)\b',
            r'\b(hai|hain|tha|thi|hoga|hogi)\b',
            r'\b(chahiye|batao|bataiye|milega|milegi)\b',
            r'\b(mujhe|mera|meri|humko|humara)\b',
            r'\b(kaisa|kaisi|konsa|konsi)\b',
        ]

        pattern_matches = 0
        for pattern in hinglish_patterns:
            if re.search(pattern, text_lower):
                pattern_matches += 1
        if pattern_matches >= 2:
            return 'hi'

        return 'en'

    # ═══════════════════════════════════════
    # 🌐 Set/Get User Language
    # ═══════════════════════════════════════
    def set_user_language(self, user_id, language):
        if language in self.supported_languages:
            self.user_language[user_id] = language
            return True
        return False

    def get_user_language(self, user_id):
        return self.user_language.get(user_id, self.default_language)

    # ═══════════════════════════════════════
    # 🌐 Get Response in Correct Language
    # ═══════════════════════════════════════
    def _get_response_for_intent(self, intent_tag, language='en'):
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                if language == 'hi':
                    hindi_responses = intent.get('responses_hi', [])
                    if hindi_responses:
                        return random.choice(hindi_responses)
                    en_responses = intent.get('responses', [])
                    if en_responses:
                        response = random.choice(en_responses)
                        return response + "\n\n_(Hindi mein jawab jaldi available hoga!)_"
                responses = intent.get('responses', [])
                if responses:
                    return random.choice(responses)

        for intent in self.intents:
            if intent['tag'] == 'default':
                if language == 'hi':
                    hindi_responses = intent.get('responses_hi', [])
                    if hindi_responses:
                        return random.choice(hindi_responses)
                return random.choice(intent.get('responses', [
                    "I'm sorry, I couldn't understand that. Please contact JECRC Foundation at +91-141-2770232."
                ]))

        return "I'm sorry, I couldn't understand that. Please contact JECRC Foundation at +91-141-2770232 for help."

    # ═══════════════════════════════════════
    # Build Vocabulary
    # ═══════════════════════════════════════
    def _build_vocabulary(self):
        self.vocabulary = set()
        for intent in self.intents:
            for pattern in intent.get('patterns', []):
                words = pattern.lower().split()
                for word in words:
                    clean = word.strip(string.punctuation)
                    if len(clean) > 2:
                        self.vocabulary.add(clean)

        important_words = {
            'admission', 'fees', 'fee', 'placement', 'hostel', 'library',
            'exam', 'result', 'syllabus', 'timetable', 'attendance',
            'scholarship', 'department', 'branch', 'course', 'college',
            'canteen', 'cafeteria', 'transport', 'bus', 'parking',
            'sports', 'wifi', 'medical', 'ragging', 'complaint',
            'grievance', 'internship', 'certificate', 'convocation',
            'alumni', 'accreditation', 'naac', 'aicte', 'eligibility',
            'backlog', 'supplementary', 'counselling', 'engineering',
            'computer', 'science', 'mechanical', 'electrical',
            'electronics', 'civil', 'jecrc', 'foundation', 'campus',
            'mess', 'faculty', 'professor', 'teacher', 'director',
            'principal', 'chairman', 'club', 'event', 'fest',
            'ncc', 'nss', 'entrepreneur', 'startup', 'research',
            'patent', 'innovation',
        }
        self.vocabulary.update(important_words)

    # ═══════════════════════════════════════
    # Levenshtein Distance
    # ═══════════════════════════════════════
    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    # ═══════════════════════════════════════
    # Fuzzy Match Word
    # ═══════════════════════════════════════
    def _fuzzy_match_word(self, word, max_distance=2):
        if not word or len(word) <= 2:
            return word, False
        word_lower = word.lower()
        if word_lower in self.vocabulary:
            return word_lower, False
        if word_lower in self.typo_corrections:
            corrected = self.typo_corrections[word_lower]
            return corrected, True

        # ════════════════════════════════════════
        # 🔧 FIX: Don't fuzzy-correct known Hindi words
        # ════════════════════════════════════════
        if word_lower in self.hindi_indicators or word_lower in self.hinglish_map:
            return word_lower, False

        best_match = None
        best_distance = float('inf')
        if len(word_lower) <= 4:
            allowed_distance = 1
        elif len(word_lower) <= 6:
            allowed_distance = 2
        else:
            allowed_distance = min(max_distance, 3)

        for vocab_word in self.vocabulary:
            if abs(len(vocab_word) - len(word_lower)) > allowed_distance:
                continue
            if word_lower[0] == vocab_word[0] or word_lower[-1] == vocab_word[-1]:
                dist = self._levenshtein_distance(word_lower, vocab_word)
                if dist < best_distance and dist <= allowed_distance:
                    best_distance = dist
                    best_match = vocab_word

        if best_match and best_distance > 0:
            return best_match, True

        return word_lower, False

    # ═══════════════════════════════════════
    # Fix Typos
    # ═══════════════════════════════════════
    def _fix_typos(self, text):
        if not text:
            return text, 0, [], []
        words = text.lower().strip().split()
        corrected_words = []
        corrections_made = 0
        original_list = []
        fixed_list = []

        # 🔧 Common English words that should NEVER be "corrected"
        protected_english = {
            'write', 'read', 'show', 'list', 'give', 'send',
            'display', 'print', 'open', 'close', 'find',
            'search', 'all', 'the', 'tell', 'about', 'what',
            'when', 'where', 'which', 'who', 'how', 'why',
            'can', 'will', 'would', 'should', 'could',
            'have', 'has', 'had', 'does', 'did', 'done',
            'make', 'made', 'take', 'took', 'give', 'gave',
            'name', 'year', 'first', 'last', 'next',
            'best', 'good', 'bad', 'new', 'old',
            'want', 'need', 'like', 'know', 'think',
        }

        # 🔧 Get protected faculty name words
        protected_faculty = set()
        if hasattr(self, 'faculty_db') and self.faculty_db:
            protected_faculty = getattr(self.faculty_db, 'protected_words', set())

        all_protected = protected_english | protected_faculty

        for word in words:
            clean_word = word.strip(string.punctuation)
            if len(clean_word) <= 2:
                corrected_words.append(clean_word)
                continue

            # Don't correct protected words
            if clean_word.lower() in all_protected:
                corrected_words.append(clean_word)
                continue

            corrected, was_corrected = self._fuzzy_match_word(clean_word)
            corrected_words.append(corrected)
            if was_corrected:
                corrections_made += 1
                original_list.append(clean_word)
                fixed_list.append(corrected)

        corrected_text = ' '.join(corrected_words)
        return corrected_text, corrections_made, original_list, fixed_list

    # ═══════════════════════════════════════
    # Devanagari to English Conversion
    # ═══════════════════════════════════════
    def _convert_devanagari_to_english(self, text):
        if not text:
            return text

        has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
        if not has_devanagari:
            return text

        words = text.strip().split()
        converted_words = []
        conversions_made = 0

        for word in words:
            clean_word = word.strip('।,!?.\'\"()[]{}:;')

            if clean_word in self.devanagari_keyword_map:
                english = self.devanagari_keyword_map[clean_word]
                converted_words.append(english)
                conversions_made += 1
            else:
                matched = False
                for hindi_word, english_word in self.devanagari_keyword_map.items():
                    if (clean_word.startswith(hindi_word) or
                        hindi_word.startswith(clean_word)) and \
                            len(clean_word) >= 2 and len(hindi_word) >= 2:
                        min_len = min(len(clean_word), len(hindi_word))
                        max_len = max(len(clean_word), len(hindi_word))
                        if min_len / max_len >= 0.6:
                            converted_words.append(english_word)
                            conversions_made += 1
                            matched = True
                            break
                if not matched:
                    converted_words.append(clean_word)

        result = ' '.join(converted_words)
        return result

    # ═══════════════════════════════════════
    # Gibberish Detection
    # ═══════════════════════════════════════
    def _is_gibberish(self, text):
        if not text:
            return True

        # ════════════════════════════════════════
        # 🔧 FIX: Check against intent patterns FIRST
        # If user message matches any intent pattern, it's NOT gibberish
        # ════════════════════════════════════════
        text_lower = text.lower().strip()

        for intent in self.intents:
            for pattern in intent.get('patterns', []):
                pattern_clean = pattern.lower().strip()

                # Exact match
                if text_lower == pattern_clean:
                    return False

                # Partial match (for longer messages)
                if len(text_lower) > 3 and len(pattern_clean) > 3:
                    if pattern_clean in text_lower:
                        return False

        # ════════════════════════════════════════
        # 🧠 Existing Gibberish Detection Logic
        # ════════════════════════════════════════
        words = text_lower.split()

        if len(text.strip()) <= 1:
            return True

        unknown_count = 0
        total_meaningful = 0

        for word in words:
            clean = word.strip(string.punctuation)

            if len(clean) <= 1:
                continue

            total_meaningful += 1

            is_known = (
                clean in self.vocabulary or
                clean in self.typo_corrections or
                clean in self.hinglish_map or
                clean in self.hindi_indicators or
                len(clean) <= 2
            )

            if not is_known:
                _, was_corrected = self._fuzzy_match_word(clean, max_distance=2)
                is_known = was_corrected

            if not is_known:
                if any('\u0900' <= char <= '\u097F' for char in clean):
                    is_known = True

            if not is_known:
                unknown_count += 1

        if total_meaningful == 0:
            return True

        unknown_ratio = unknown_count / total_meaningful

        if unknown_ratio > 0.7 and total_meaningful >= 2:
            return True

        # Repetitive character check (like "aaaa", "xxx")
        if len(text.strip()) > 3:
            unique_chars = len(set(text.strip().replace(' ', '')))
            if unique_chars <= 2:
                return True

        return False

    def _load_intents(self, intents_file):
        try:
            with open(intents_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.intents = data['intents']
            print(f"✅ Loaded {len(self.intents)} intents from {intents_file}")
            hindi_count = sum(1 for i in self.intents if i.get('responses_hi'))
            print(f"  🌐 Hindi responses available: {hindi_count}/{len(self.intents)} intents")
        except FileNotFoundError:
            print(f"❌ Error: {intents_file} not found!")
            self.intents = []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            self.intents = []

    def _preprocess(self, text):
        """Preprocess user input text"""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)

        # Convert Devanagari Hindi to English keywords FIRST
        text = self._convert_devanagari_to_english(text)

        text, num_fixes, _, _ = self._fix_typos(text)
        words = text.split()
        translated_words = []
        for word in words:
            clean_word = word.strip(string.punctuation)
            if clean_word in self.hinglish_map:
                translated_words.append(self.hinglish_map[clean_word])
            else:
                translated_words.append(clean_word)
        text = ' '.join(translated_words)
        text = text.translate(str.maketrans('', '', string.punctuation))
        try:
            tokens = word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens if len(token) > 1
            ]
            return ' '.join(tokens)
        except Exception:
            return text

    def _train(self):
        """Train TF-IDF model on all intent patterns"""
        if not self.intents:
            print("❌ No intents to train on!")
            return
        self.all_patterns = []
        self.pattern_intent_map = []
        for intent in self.intents:
            tag = intent.get('tag', '')
            patterns = intent.get('patterns', [])
            for pattern in patterns:
                processed = self._preprocess(pattern)
                if processed:
                    self.all_patterns.append(processed)
                    self.pattern_intent_map.append(tag)
        if self.all_patterns:
            self.intent_vectors = self.vectorizer.fit_transform(self.all_patterns)
            print(f"✅ Trained on {len(self.all_patterns)} patterns across {len(self.intents)} intents")

            # 🔧 NEW: Auto-build keyword map from patterns
            self._build_auto_keywords()

        else:
            print("❌ No valid patterns found for training!")

    # ═══════════════════════════════════════
    # 🔧 NEW: Auto Keyword Builder
    # ═══════════════════════════════════════
    def _build_auto_keywords(self):
        """
        Automatically extract distinctive keywords from intent patterns.
        No more manual keyword adding needed!
        """
        # Step 1: Count which words appear in which intents
        word_intent_count = {}   # word → set of intent tags
        word_intent_freq = {}    # word → {intent: frequency}

        # Common stop words to skip
        stop_words = {
            "english", "hindi", "hinglish", "language",
            "bhasha", "bolte", "baat", "speak",
            "mein", "me", "in", "ke", "ka", "ki",
            "hai", "ho", "hain", "kya", "kaise",
            'i', 'me', 'my', 'we', 'you', 'your', 'he', 'she', 'it',
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into',
            'about', 'like', 'after', 'before', 'between',
            'do', 'does', 'did', 'have', 'has', 'had',
            'can', 'could', 'will', 'would', 'shall', 'should',
            'may', 'might', 'must',
            'not', 'no', 'nor', 'so', 'if', 'then', 'than',
            'very', 'just', 'also', 'more', 'most', 'much',
            'what', 'which', 'who', 'whom', 'this', 'that',
            'dean', 'deans', 'mam', 'maam', 'madam',
            'these', 'those', 'how', 'when', 'where', 'why',
            'all', 'each', 'every', 'both', 'few', 'some', 'any',
            'tell', 'know', 'get', 'give', 'make', 'go', 'come',
            'want', 'need', 'take', 'see', 'look', 'write',
            'read', 'show', 'list', 'give', 'send',
            'display', 'print', 'open', 'close', 'find',
            'search', 'noite', 'nuit', 'nacht', 'notte',
            # Hindi stop words
            'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'ne',
            'par', 'pe', 'mein', 'ye', 'woh', 'yeh', 'wo',
            'toh', 'bhi', 'aur', 'ya', 'kya', 'kaise', 'kab',
            'kahan', 'kaun', 'konsa', 'konsi', 'kitna', 'kitni',
            'kitne', 'mujhe', 'mera', 'meri', 'mere', 'hum',
            'aap', 'tum', 'tu', 'mai', 'main', 'tera', 'teri',
            'uska', 'uski', 'unka', 'batao', 'bataiye', 'bataye',
            'chahiye', 'hoga', 'hogi', 'karo', 'karna', 'karke',
            'milega', 'milegi', 'milta', 'milti', 'i', 'me', 'my', 'we', 'you', 'your', 'he', 'she', 'it',
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into',
            'about', 'like', 'after', 'before', 'between',
            'do', 'does', 'did', 'have', 'has', 'had',
            'can', 'could', 'will', 'would', 'shall', 'should',
            'may', 'might', 'must',
            'not', 'no', 'nor', 'so', 'if', 'then', 'than',
            'very', 'just', 'also', 'more', 'most', 'much',
            'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'how', 'when', 'where', 'why',
            'all', 'each', 'every', 'both', 'few', 'some', 'any',
            'tell', 'know', 'get', 'give', 'make', 'go', 'come',
            'want', 'need', 'take', 'see', 'look',
            'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'ne',
            'par', 'pe', 'mein', 'ye', 'woh', 'yeh', 'wo',
            'toh', 'bhi', 'aur', 'ya', 'kya', 'kaise', 'kab',
            'kahan', 'kaun', 'konsa', 'konsi', 'kitna', 'kitni',
            'kitne', 'mujhe', 'mera', 'meri', 'mere', 'hum',
            'aap', 'tum', 'tu', 'mai', 'main', 'tera', 'teri',
            'uska', 'uski', 'unka', 'batao', 'bataiye', 'bataye',
            'chahiye', 'hoga', 'hogi', 'karo', 'karna', 'karke',
            'milega', 'milegi', 'milta', 'milti',
            # 🔧 NEW: More generic Hindi words (not distinctive keywords)
            'kaam', 'kaam', 'aapka', 'aapke', 'aapki', 'aapko',
            'tumhara', 'tumhari', 'tumhare', 'tumhe', 'tumko',
            'inka', 'inki', 'inke', 'unki', 'unke',
            'hamara', 'hamari', 'hamare', 'humko', 'humne',
            'iska', 'iski', 'iske', 'jiska', 'jiski', 'jiske',
            'karta', 'karti', 'karte', 'kiya', 'kiye', 'karenge',
            'karega', 'karegi', 'karoge', 'karunga', 'karungi',
            'hota', 'hoti', 'hote', 'hua', 'hui', 'hue',
            'raha', 'rahi', 'rahe', 'rahega', 'rahegi', 'rahenge',
            'sakta', 'sakti', 'sakte', 'pata', 'sahi', 'galat',
            'accha', 'achi', 'ache', 'bura', 'buri', 'bure',
            'naya', 'nayi', 'naye', 'purana', 'purani', 'purane',
            'bada', 'badi', 'bade', 'chota', 'choti', 'chote',
            'yahan', 'wahan', 'idhar', 'udhar', 'jahan',
            'abhi', 'pehle', 'baad', 'phir', 'tab', 'jab',
            'bahut', 'thoda', 'zyada', 'kam', 'sab', 'koi',
            'kuch', 'kahin', 'kabhi', 'hamesha', 'jaruri',
            'zaruri', 'jarurat', 'zarurat',
            'wala', 'wali', 'wale', 'waala', 'waali', 'waale',
            'liye', 'saath', 'sath', 'bina', 'sirf', 'dono',
            'teeno', 'chaaro', 'log', 'banda', 'bande',
            'sir', 'madam', 'bhai', 'bro', 'dost', 'yaar',
            'please', 'plz', 'pls', 'thanku', 'okay', 'okk',
            'yes', 'yeah', 'yep', 'haan', 'nahi', 'nhi', 'naah',
            'good', 'bad', 'nice', 'great', 'best', 'worst'
        }

        for intent in self.intents:
            tag = intent.get('tag', '')
            if tag in ['default', 'unclear_retype']:
                continue
            patterns = intent.get('patterns', [])
            for pattern in patterns:
                words = pattern.lower().strip().split()
                for word in words:
                    clean = word.strip(string.punctuation).lower()
                    if len(clean) <= 2 or clean in stop_words:
                        continue

                    if clean not in word_intent_count:
                        word_intent_count[clean] = set()
                        word_intent_freq[clean] = {}

                    word_intent_count[clean].add(tag)
                    word_intent_freq[clean][tag] = word_intent_freq[clean].get(tag, 0) + 1

        # Step 2: Build auto keywords
        # A word is a good keyword if it appears in only 1-3 intents
        self.auto_keywords = {}

        for word, intent_set in word_intent_count.items():
            if len(intent_set) <= 3:
                # Pick the intent where this word appears most frequently
                best_intent = max(
                    word_intent_freq[word],
                    key=word_intent_freq[word].get
                )
                best_freq = word_intent_freq[word][best_intent]

                # Only add if word appears at least 2 times in that intent
                # OR if it appears in only 1 intent (very distinctive)
                if best_freq >= 2 or len(intent_set) == 1:
                    self.auto_keywords[word] = best_intent

        print(f"  🔑 Auto-keywords built: {len(self.auto_keywords)} keywords")

    def _classify_intent(self, user_message):
        """Classify user message using TF-IDF + Cosine Similarity"""
        processed_msg = self._preprocess(user_message)
        if not processed_msg or self.intent_vectors is None:
            return 'default', 0.0
        try:
            user_vector = self.vectorizer.transform([processed_msg])
            similarities = cosine_similarity(user_vector, self.intent_vectors).flatten()
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            best_intent = self.pattern_intent_map[best_idx]

            top_indices = np.argsort(similarities)[-3:][::-1]
            top_intents = [self.pattern_intent_map[i] for i in top_indices]
            top_scores = [similarities[i] for i in top_indices]

            if len(top_intents) >= 2 and top_intents[0] == top_intents[1]:
                best_score = min(best_score * 1.2, 1.0)

            print(f"    📊 TF-IDF Top 3 for '{user_message}':")
            for i in range(min(3, len(top_indices))):
                print(f"      {i+1}. {top_intents[i]} ({top_scores[i]:.2%})")

            return best_intent, float(best_score)
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return 'default', 0.0

    def _keyword_fallback(self, user_message):
        """Fallback: Match keywords when TF-IDF confidence is low"""
        msg_lower = user_message.lower().strip()
        corrected_msg, _, _, _ = self._fix_typos(msg_lower)

        keyword_map = {
            # In _keyword_fallback method, keyword_map dictionary:
            'namaste': 'greeting_indian_cultural',
            'ram ram': 'greeting_indian_cultural',
            'kaise ho': 'greeting_how_are_you',
            'kya haal': 'greeting_how_are_you',
            'wassup': 'greeting_casual_slang',
            'whats up': 'greeting_casual_slang',
            'good morning': 'greeting_time_based',
            'good evening': 'greeting_time_based',
            'alvida': 'goodbye_hindi_farewell',
            'phir milenge': 'goodbye_hindi_farewell',
            'dhanyawad': 'thanks_hindi',
            'shukriya': 'thanks_hindi',
            'research center': 'research_publications',
            'lateral entry': 'admission_lateral',
            'diploma to btech': 'admission_lateral',
            'reap counselling': 'reap_counselling',
            'jee main': 'jee_main', 'jee score': 'jee_main',
            'gap year': 'gap_year', 'gap certificate': 'gap_year',
            'last date': 'admission_deadline',
            'ai ml': 'aiml_specialization',
            'artificial intelligence': 'aiml_specialization',
            'machine learning': 'aiml_specialization',
            'data science': 'data_science_specialization',
            'cyber security': 'cyber_security_specialization',
            'computer science': 'cse_department',
            'information technology': 'it_department',
            'civil engineering': 'ce_department',
            'seminar hall': 'auditorium_seminar',
            'campus life': 'campus_life',
            'how to reach': 'campus_navigation',
            'student portal': 'student_login_erp',
            'student login': 'student_login_erp',
            'green campus': 'environment_green',
            'near college': 'nearby_places',
            'id card': 'dress_code', 'dress code': 'dress_code',
            'back paper': 'backlog_kt',
            'semester system': 'semester_system',
            'higher studies': 'higher_studies',
            'after btech': 'higher_studies',
            'branch wise placement': 'placement_statistics_detailed',
            'placement training': 'placement_training',
            'fee refund': 'fee_refund',
            'fee payment': 'fee_payment',
            'how to pay': 'fee_payment',
            'summer training': 'internship',
            'who are you': 'bot_identity',
            'about college': 'about_college',
            'about jecrc': 'about_college',
            'jecrc foundation': 'about_college',
            'is jecrc good': 'admission_comparison',
            'anti ragging': 'ragging',
            'women cell': 'women_cell',
            'mental health': 'counselling_mental_health',
            'anti drug': 'anti_drug',
            'reap': 'reap_counselling',
            'eligib': 'admission_eligibility',
            'cutoff': 'admission_eligibility',
            'document': 'admission_documents',
            'deadline': 'admission_deadline',
            'admission': 'admission_process',
            'apply': 'admission_process',
            'cse': 'cse_department', 'ece': 'ece_department',
            'electrical': 'ee_department',
            'mechanical': 'me_department',
            'mba': 'mba_program', 'mca': 'mca_program',
            'mtech': 'mtech_program',
            'department': 'departments', 'branch': 'departments',
            'course': 'departments',
            'scholarship': 'scholarship',
            'refund': 'fee_refund',
            'installment': 'fee_payment',
            'fee': 'fee_structure', 'fees': 'fee_structure',
            'placement': 'placement', 'package': 'placement',
            'internship': 'internship',
            'hostel rule': 'hostel_rules',
            'mess': 'mess_food', 'canteen': 'mess_food',
            'food': 'mess_food', 'cafeteria': 'mess_food',
            'hostel': 'hostel',
            'backlog': 'backlog_kt', 'supplementary': 'backlog_kt',
            'attendance': 'attendance',
            'result': 'result', 'marks': 'result',
            'cgpa': 'result', 'sgpa': 'result',
            'exam': 'exam_schedule',
            'syllabus': 'syllabus',
            'timetable': 'timetable', 'schedule': 'timetable',
            'faculty': 'hod_faculty', 'hod': 'hod_faculty',
            'professor': 'hod_faculty', 'teacher': 'hod_faculty',
            'gate': 'higher_studies',
            'certification': 'certifications_courses',
            'nptel': 'certifications_courses',
            'transfer': 'transfer_migration',
            'convocation': 'convocation', 'degree': 'convocation',
            'library': 'library', 'book': 'library',
            'lab': 'lab_facilities',
            'sport': 'sports', 'gym': 'sports',
            'wifi': 'wifi', 'internet': 'wifi',
            'bus': 'transport', 'transport': 'transport',
            'parking': 'parking',
            'medical': 'medical', 'doctor': 'medical',
            'auditorium': 'auditorium_seminar',
            'campus': 'campus_navigation',
            'location': 'campus_navigation',
            'erp': 'student_login_erp',
            'ragging': 'ragging',
            'icc': 'women_cell',
            'stress': 'counselling_mental_health',
            'counselling': 'counselling_mental_health',
            'drug': 'anti_drug', 'smoking': 'anti_drug',
            'complaint': 'complaint', 'grievance': 'complaint',
            'discipline': 'discipline_rules', 'rules': 'discipline_rules',
            'disability': 'disability_support',
            'chairman': 'chairman_management',
            'director': 'director_principal',
            'principal': 'director_principal',
            'vision': 'vision_mission', 'mission': 'vision_mission',
            'accreditation': 'accreditation', 'naac': 'accreditation',
            'nba': 'accreditation', 'aicte': 'accreditation',
            'alumni': 'alumni',
            'event': 'events', 'fest': 'events',
            'club': 'clubs', 'society': 'clubs',
            'nss': 'ncc_nss', 'ncc': 'ncc_nss',
            'entrepreneur': 'entrepreneurship_ecell',
            'startup': 'entrepreneurship_ecell',
            'research': 'research_publications',
            'uniform': 'dress_code',
            'parent': 'parent_guardian', 'ptm': 'parent_guardian',
            'contact': 'contact', 'phone': 'contact',
            'email': 'contact',
            'namaste': 'greeting', 'hello': 'greeting',
            'hey': 'greeting', 'hi': 'greeting',
            'bye': 'goodbye', 'goodbye': 'goodbye',
            'thank': 'thanks',
            'help': 'help',
            'appreciate': 'thanks_appreciation',
            'appreciated': 'thanks_appreciation',
            'great job': 'thanks_appreciation',
            'well done': 'thanks_appreciation',
            'helpful': 'thanks_appreciation',
            'amazing': 'thanks_appreciation',
            'awesome': 'thanks_appreciation',
            'perfect': 'thanks_appreciation',
            'wonderful': 'thanks_appreciation',
            'fantastic': 'thanks_appreciation',
            'maza aa gaya': 'thanks_appreciation',
            'bahut accha': 'thanks_appreciation',
            'very nice': 'thanks_appreciation',
            'good job': 'thanks_appreciation',
        }

        converted_msg = self._convert_devanagari_to_english(msg_lower)

        for text_to_check in [msg_lower, corrected_msg, converted_msg]:
            for keyword, intent_tag in keyword_map.items():
                if keyword in text_to_check:
                    print(f"    🔑 Keyword matched: '{keyword}' → {intent_tag}")
                    return intent_tag, keyword
    def _keyword_fallback(self, user_message):
        """Fallback: Match keywords when TF-IDF confidence is low"""
        msg_lower = user_message.lower().strip()
        corrected_msg, _, _, _ = self._fix_typos(msg_lower)

        keyword_map = {
            # ... existing keyword_map stays exactly same ...
            'help': 'help',
        }

        converted_msg = self._convert_devanagari_to_english(msg_lower)

        for text_to_check in [msg_lower, corrected_msg, converted_msg]:
            for keyword, intent_tag in keyword_map.items():
                if keyword in text_to_check:
                    print(f"    🔑 Keyword matched: '{keyword}' → {intent_tag}")
                    return intent_tag, keyword

        # ════════════════════════════════════════
        # 🔧 NEW: Auto-keyword fallback
        # Check against auto-extracted keywords from patterns
        # ════════════════════════════════════════
        if hasattr(self, 'auto_keywords') and self.auto_keywords:
            best_auto_intent = None
            best_auto_keyword = None
            best_auto_score = 0

            for text_to_check in [msg_lower, corrected_msg, converted_msg]:
                words = text_to_check.split()
                # Count how many auto-keywords match per intent
                intent_match_count = {}
                intent_matched_words = {}

                for word in words:
                    clean_word = word.strip(string.punctuation)
                    if clean_word in self.auto_keywords:
                        matched_intent = self.auto_keywords[clean_word]
                        intent_match_count[matched_intent] = intent_match_count.get(matched_intent, 0) + 1
                        if matched_intent not in intent_matched_words:
                            intent_matched_words[matched_intent] = []
                        intent_matched_words[matched_intent].append(clean_word)

                # Also check 2-word phrases (bigrams)
                for i in range(len(words) - 1):
                    bigram = f"{words[i].strip(string.punctuation)} {words[i+1].strip(string.punctuation)}"
                    if bigram in self.auto_keywords:
                        matched_intent = self.auto_keywords[bigram]
                        intent_match_count[matched_intent] = intent_match_count.get(matched_intent, 0) + 2
                        if matched_intent not in intent_matched_words:
                            intent_matched_words[matched_intent] = []
                        intent_matched_words[matched_intent].append(bigram)

                # Find best matching intent
                for intent_tag, count in intent_match_count.items():
                    if count > best_auto_score:
                        best_auto_score = count
                        best_auto_intent = intent_tag
                        best_auto_keyword = ', '.join(intent_matched_words[intent_tag])

            if best_auto_intent and best_auto_score >= 1:
                print(f"    🔑 Auto-keyword matched: '{best_auto_keyword}' → {best_auto_intent} (score: {best_auto_score})")
                return best_auto_intent, best_auto_keyword

        return None, None

    def _exact_match(self, user_message):
        """Check if user message exactly matches any pattern"""
        msg_lower = user_message.lower().strip()
        corrected_msg, _, _, _ = self._fix_typos(msg_lower)

        for check_msg in [msg_lower, corrected_msg]:
            for intent in self.intents:
                tag = intent.get('tag', '')
                patterns = intent.get('patterns', [])
                for pattern in patterns:
                    pattern_lower = pattern.lower().strip()
                    if check_msg == pattern_lower:
                        return tag, 1.0
                    if check_msg in pattern_lower or pattern_lower in check_msg:
                        if len(check_msg) > 3 and len(pattern_lower) > 3:
                            overlap = min(len(check_msg), len(pattern_lower)) / max(len(check_msg), len(pattern_lower))
                            if overlap > 0.6:
                                return tag, 0.85
        return None, 0

    # ═══════════════════════════════════════
    # 🌐 Generate Retype Message (Multi-Language)
    # ═══════════════════════════════════════
    def _generate_retype_message(self, original_msg, corrected_msg, num_corrections, language='en'):
        """Generate a helpful retype message in the correct language"""

        if language == 'hi':
            retype_messages_hi = [
                (
                    "माफ़ कीजिए, मैं आपका सवाल समझ नहीं पाया। 🤔\n\n"
                    "कृपया अपना सवाल **दोबारा टाइप** करें!\n\n"
                    "आप इनके बारे में पूछ सकते हैं:\n"
                    "🔹 एडमिशन प्रक्रिया\n"
                    "🔹 फीस स्ट्रक्चर\n"
                    "🔹 प्लेसमेंट डिटेल्स\n"
                    "🔹 हॉस्टल सुविधाएं\n"
                    "🔹 परीक्षा और सिलेबस\n\n"
                    "💡 उदाहरण: \"admission kaise hota hai?\", \"fees kitni hai?\""
                ),
                (
                    "मैं आपकी बात समझ नहीं पाया। 😅\n\n"
                    "कृपया **सरल शब्दों** में दोबारा पूछें!\n\n"
                    "📌 जैसे:\n"
                    "• \"एडमिशन कैसे होता है?\"\n"
                    "• \"फीस कितनी है?\"\n"
                    "• \"हॉस्टल के बारे में बताओ\"\n"
                    "• \"प्लेसमेंट कैसी है?\"\n\n"
                    "या **\"help\"** टाइप करें सभी विषय देखने के लिए! 📋"
                ),
            ]

            if num_corrections > 0:
                return (
                    f"मैंने आपका सवाल समझने की कोशिश की लेकिन पूरी तरह समझ नहीं पाया। 🤔\n\n"
                    f"कृपया **दोबारा ध्यान से टाइप** करें!\n\n"
                    f"💡 **सुझाव:**\n"
                    f"🔹 सरल हिंदी या English में लिखें\n"
                    f"🔹 जैसे: \"admission kaise hota hai?\"\n"
                    f"🔹 जैसे: \"fees kitni hai?\"\n\n"
                    f"या साइडबार से कोई टॉपिक चुनें! 👈"
                )

            return random.choice(retype_messages_hi)

        # English retype messages
        retype_messages = [
            (
                "I'm not sure I understood that correctly. 🤔\n\n"
                "Could you please **retype your question** a bit more clearly?\n\n"
                "Here are some things I can help with:\n"
                "🔹 Admissions & Eligibility\n"
                "🔹 Fee Structure & Scholarships\n"
                "🔹 Placements & Training\n"
                "🔹 Hostel & Campus Life\n"
                "🔹 Exams, Results & Syllabus\n\n"
                "💡 **Tip:** Try typing in simple words like:\n"
                "\"admission process\", \"hostel fees\", \"placement details\""
            ),
            (
                "Hmm, I couldn't quite catch that. 😅\n\n"
                "Could you **please type your question again** in a simpler way?\n\n"
                "📌 For example:\n"
                "• \"How to get admission?\"\n"
                "• \"What is the fee structure?\"\n"
                "• \"Tell me about hostel\"\n"
                "• \"Placement details batao\"\n\n"
                "You can also use the **Quick Topics** on the left sidebar! 👈"
            ),
            (
                "I didn't quite understand your question. 🙁\n\n"
                "Please **try rephrasing** or check for any typos!\n\n"
                "🎯 I'm best at answering about:\n"
                "📋 Admissions | 💰 Fees | 💼 Placements\n"
                "🏠 Hostel | 📚 Academics | 🏛️ Campus Life\n\n"
                "Or type **\"help\"** to see all available topics!"
            ),
        ]

        if num_corrections > 0:
            return (
                f"I tried to understand your question but I'm still not sure what you're looking for. 🤔\n\n"
                f"Could you please **type your question again more carefully**?\n\n"
                f"💡 **Tips:**\n"
                f"🔹 Use simple English or Hinglish\n"
                f"🔹 Example: \"admission kaise hota hai?\"\n"
                f"🔹 Example: \"fees kitni hai?\"\n"
                f"🔹 Example: \"placement details\"\n\n"
                f"Or click a topic from the sidebar for quick answers! 👈"
            )

        return random.choice(retype_messages)


    def get_response(self, user_message, user_id=None, language=None):
        """Main response method"""
        print(f"\n{'='*60}")
        print(f"  📝 User: '{user_message}'")
        print(f"{'='*60}")

        # STEP 1: Language Detection
        current_lang = self.detect_language(user_message)
        print(f"  🌐 Language: {current_lang}")

        # STEP 2: Gibberish Detection
        if self._is_gibberish(user_message):
            response = self._generate_retype_message(user_message, user_message, 0, current_lang)
            print(f"  ✅ Result: gibberish detected")
            return {
                'reply': response,
                'intent': 'gibberish',
                'confidence': 0.0,
                'method': 'gibberish_detection',
                'language': current_lang
            }

        # STEP 3: Faculty Search
        if self.faculty_search_enabled and self.faculty_db:
            print(f"  🔍 Checking faculty search...")
            faculty_result = self._try_faculty_search(user_message, current_lang)
            if faculty_result:
                print(f"  ✅ Faculty HIT: {faculty_result.get('intent', '')}")
                return {
                    'reply': faculty_result.get('reply', ''),
                    'intent': faculty_result.get('intent', 'faculty'),
                    'confidence': faculty_result.get('confidence', 0.95),
                    'method': 'faculty_db_search',
                    'language': current_lang
                }
            else:
                print(f"  ❌ Faculty search: No match")

        # STEP 4: Fix Typos
        corrected_text, num_corrections, original_words, fixed_words = self._fix_typos(user_message)
        if num_corrections > 0:
            print(f"  🔤 Typos fixed ({num_corrections}): {list(zip(original_words, fixed_words))}")

        # STEP 5: Multi-Intent Detection
        multi_intents = self._detect_multi_intent(
            corrected_text if num_corrections > 0 else user_message,
            current_lang
        )

        if multi_intents and len(multi_intents) >= 2:
            combined_response = ""
            for i, intent_tag in enumerate(multi_intents):
                response = self._get_response_for_intent(intent_tag, current_lang)
                if i > 0:
                    combined_response += "\n\n━━━━━━━━━━━━━━━━━━━━\n\n"
                combined_response += response

            intent_str = ' + '.join(multi_intents)
            print(f"  ✅ Multi-intent: {intent_str}")
            return {
                'reply': combined_response,
                'intent': intent_str,
                'confidence': 0.85,
                'method': 'multi_intent',
                'language': current_lang
            }

        # STEP 6: Exact Match
        exact_intent, exact_conf = self._exact_match(user_message)
        if exact_intent and exact_conf >= 0.85:
            response = self._get_response_for_intent(exact_intent, current_lang)
            print(f"  ✅ Result: {exact_intent} (exact, {exact_conf:.0%})")
            return {
                'reply': response,
                'intent': exact_intent,
                'confidence': round(exact_conf, 4),
                'method': 'exact',
                'language': current_lang
            }

        # STEP 7: TF-IDF + Keyword
        tfidf_intent, tfidf_conf = self._classify_intent(
            corrected_text if num_corrections > 0 else user_message
        )
        keyword_intent, keyword_matched = self._keyword_fallback(
            corrected_text if num_corrections > 0 else user_message
        )

        if num_corrections > 0 and tfidf_conf < self.confidence_threshold:
            tfidf_intent_orig, tfidf_conf_orig = self._classify_intent(user_message)
            if tfidf_conf_orig > tfidf_conf:
                tfidf_intent = tfidf_intent_orig
                tfidf_conf = tfidf_conf_orig

        print(f"  📊 TF-IDF: {tfidf_intent} ({tfidf_conf:.2%})")
        print(f"  🔑 Keyword: {keyword_intent} (matched: '{keyword_matched}')")

        # STEP 8: Smart Decision
        final_intent = 'default'
        final_conf = 0.0
        method = 'default'

        if keyword_intent and tfidf_intent == keyword_intent:
            final_intent = tfidf_intent
            final_conf = min(tfidf_conf + 0.15, 1.0)
            method = 'hybrid'
        elif tfidf_conf >= self.high_confidence_threshold:
            final_intent = tfidf_intent
            final_conf = tfidf_conf
            method = 'tfidf'
        elif keyword_intent and tfidf_conf < self.high_confidence_threshold:
            final_intent = keyword_intent
            final_conf = 0.70
            method = 'keyword'
        elif tfidf_conf >= self.confidence_threshold and not keyword_intent:
            final_intent = tfidf_intent
            final_conf = tfidf_conf
            method = 'tfidf'
        else:
            final_intent = 'unclear'
            final_conf = max(tfidf_conf, 0.0)
            method = 'unclear'

        # STEP 9: Retype if no good match
        if final_intent in ['default', 'unclear']:
            response = self._generate_retype_message(
                user_message, corrected_text, num_corrections, current_lang
            )
            print(f"  ✅ Final: retype_suggestion | {final_conf:.2%}")
            return {
                'reply': response,
                'intent': 'unclear_retype',
                'confidence': round(final_conf, 4),
                'method': 'retype_suggestion',
                'language': current_lang
            }

        # STEP 10: Final Response
        response = self._get_response_for_intent(final_intent, current_lang)
        print(f"  ✅ Final: {final_intent} | {final_conf:.2%} | {method}")
        return {
            'reply': response,
            'intent': final_intent,
            'confidence': round(final_conf, 4),
            'method': method,
            'language': current_lang
        }

    def get_all_intents(self):
        return [intent['tag'] for intent in self.intents if intent['tag'] != 'default']

    def get_intent_count(self):
        return len(self.intents)

    def get_pattern_count(self):
        return len(self.all_patterns)

    def get_stats(self):
        hindi_count = sum(1 for i in self.intents if i.get('responses_hi'))
        return {
            'total_intents': self.get_intent_count(),
            'total_patterns': self.get_pattern_count(),
            'confidence_threshold': self.confidence_threshold,
            'high_confidence_threshold': self.high_confidence_threshold,
            'typo_corrections_loaded': len(self.typo_corrections),
            'vocabulary_size': len(self.vocabulary),
            'supported_languages': self.supported_languages,
            'hindi_responses_available': hindi_count,
            'intents_list': self.get_all_intents()
        }

# ── Testing ──
if __name__ == "__main__":
    print("=" * 60)
    print("  JECRC Foundation Helpdesk AI - Engine Test")
    print("  🌐 WITH MULTI-LANGUAGE SUPPORT")
    print("  🤖 WITH CONTEXT-AWARE FOLLOW-UPS")
    print("=" * 60)

    engine = ChatbotEngine()
    stats = engine.get_stats()
    print(f"\n📊 Stats:")
    print(f"  Intents: {stats['total_intents']}")
    print(f"  Patterns: {stats['total_patterns']}")
    print(f"  Languages: {stats['supported_languages']}")
    print(f"  Hindi Responses: {stats['hindi_responses_available']}")

