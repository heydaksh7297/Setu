"""
============================================================
  JECRC Foundation - College Helpdesk AI Chatbot
  Database Handler - SQLite
  Stores chat history, analytics, feedback, unresolved queries
  
  📥 NEW: get_export_data() for CSV/PDF export with filters
============================================================
"""

import sqlite3
import datetime
from contextlib import contextmanager


class ChatDatabase:
    """SQLite Database Manager for Chat History & Analytics"""

    def __init__(self, db_path='chat_history.db'):
        self.db_path = db_path
        self._create_tables()
        print(f"✅ Database initialized: {db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for safe database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"❌ Database error: {e}")
            raise e
        finally:
            conn.close()

    def _create_tables(self):
        """Create all necessary database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Chat History Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    intent TEXT DEFAULT 'unknown',
                    confidence REAL DEFAULT 0.0,
                    method TEXT DEFAULT 'unknown',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT DEFAULT '',
                    user_agent TEXT DEFAULT ''
                )
            ''')

            # Feedback Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                    comment TEXT DEFAULT '',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chat_history(id)
                )
            ''')

            # Unresolved Queries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unresolved_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    session_id TEXT,
                    confidence REAL DEFAULT 0.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0,
                    admin_response TEXT DEFAULT ''
                )
            ''')

    # ✅ Cleanup old chats
    def cleanup_old_chats(self, days=90):
        """Delete chat history older than X days"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM chat_history 
                WHERE timestamp < datetime('now', ? || ' days')
            ''', (f'-{days}',))
            deleted = cursor.rowcount
            print(f"🗑️ Cleaned up {deleted} old chat records (>{days} days)")
            return deleted

    def save_chat(self, session_id, user_message, bot_response,
                  intent='unknown', confidence=0.0, method='unknown',
                  ip_address='', user_agent=''):
        """Save a chat interaction to database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_history
                (session_id, user_message, bot_response, intent, confidence, method, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, user_message, bot_response, intent,
                  confidence, method, ip_address, user_agent))

            chat_id = cursor.lastrowid

            # Save as unresolved if low confidence
            if confidence < 0.35 and intent == 'default':
                cursor.execute('''
                    INSERT INTO unresolved_queries (user_message, session_id, confidence)
                    VALUES (?, ?, ?)
                ''', (user_message, session_id, confidence))

            return chat_id

    def save_feedback(self, chat_id, rating, comment=''):
        """Save user feedback"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO feedback (chat_id, rating, comment)
                VALUES (?, ?, ?)
            ''', (chat_id, rating, comment))

    def get_chat_history(self, session_id=None, limit=50):
        """Get chat history"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute('''
                    SELECT * FROM chat_history
                    WHERE session_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (session_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM chat_history
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_analytics(self):
        """Get dashboard analytics data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) as total FROM chat_history')
            total_chats = cursor.fetchone()['total']

            today = datetime.date.today().isoformat()
            cursor.execute(
                'SELECT COUNT(*) as count FROM chat_history WHERE DATE(timestamp) = ?',
                (today,)
            )
            today_chats = cursor.fetchone()['count']

            cursor.execute('SELECT COUNT(DISTINCT session_id) as count FROM chat_history')
            unique_sessions = cursor.fetchone()['count']

            cursor.execute('SELECT AVG(confidence) as avg FROM chat_history WHERE confidence > 0')
            avg_confidence = cursor.fetchone()['avg'] or 0

            cursor.execute('''
                SELECT intent, COUNT(*) as count
                FROM chat_history
                WHERE intent NOT IN ('default', 'greeting', 'goodbye', 'thanks', 'empty')
                GROUP BY intent
                ORDER BY count DESC
                LIMIT 10
            ''')
            top_intents = [dict(row) for row in cursor.fetchall()]

            cursor.execute('SELECT COUNT(*) as count FROM unresolved_queries WHERE resolved = 0')
            unresolved = cursor.fetchone()['count']

            cursor.execute('''
                SELECT * FROM unresolved_queries
                WHERE resolved = 0
                ORDER BY timestamp DESC LIMIT 20
            ''')
            unresolved_queries = [dict(row) for row in cursor.fetchall()]

            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM chat_history
                GROUP BY hour
                ORDER BY hour
            ''')
            hourly_dist = [dict(row) for row in cursor.fetchall()]

            cursor.execute('''
                SELECT method, COUNT(*) as count
                FROM chat_history
                GROUP BY method
            ''')
            method_dist = [dict(row) for row in cursor.fetchall()]

            return {
                'total_chats': total_chats,
                'today_chats': today_chats,
                'unique_sessions': unique_sessions,
                'avg_confidence': round(avg_confidence, 4),
                'top_intents': top_intents,
                'unresolved_count': unresolved,
                'unresolved_queries': unresolved_queries,
                'hourly_distribution': hourly_dist,
                'method_distribution': method_dist
            }

    def resolve_query(self, query_id, admin_response=''):
        """Mark an unresolved query as resolved"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE unresolved_queries
                SET resolved = 1, admin_response = ?
                WHERE id = ?
            ''', (admin_response, query_id))

    def clear_history(self):
        """Clear all chat history (admin function)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM chat_history')
            cursor.execute('DELETE FROM feedback')
            cursor.execute('DELETE FROM unresolved_queries')
            print("✅ All chat history cleared!")

    # ══════════════════════════════════════
    # 📊 Advanced Analytics for Dashboard
    # ══════════════════════════════════════

    def get_daily_chat_counts(self, days=30):
        """Get chat count per day for last N days"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM chat_history
                WHERE timestamp >= datetime('now', ? || ' days')
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            ''', (f'-{days}',))
            return [dict(row) for row in cursor.fetchall()]

    def get_hourly_distribution(self):
        """Get chat count per hour (0-23)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                       COUNT(*) as count
                FROM chat_history
                GROUP BY hour
                ORDER BY hour
            ''')
            # Fill missing hours with 0
            result = {i: 0 for i in range(24)}
            for row in cursor.fetchall():
                result[row['hour']] = row['count']
            return [{'hour': h, 'count': c} for h, c in result.items()]

    def get_method_distribution(self):
        """Get distribution of classification methods"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT method, COUNT(*) as count
                FROM chat_history
                WHERE method IS NOT NULL AND method != ''
                GROUP BY method
                ORDER BY count DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_confidence_trends(self, days=7):
        """Get average confidence per day"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DATE(timestamp) as date,
                       ROUND(AVG(confidence), 4) as avg_confidence,
                       ROUND(MIN(confidence), 4) as min_confidence,
                       ROUND(MAX(confidence), 4) as max_confidence,
                       COUNT(*) as total
                FROM chat_history
                WHERE confidence > 0
                  AND timestamp >= datetime('now', ? || ' days')
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            ''', (f'-{days}',))
            return [dict(row) for row in cursor.fetchall()]

    def get_top_intents(self, limit=10):
        """Get most common intents"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT intent, COUNT(*) as count,
                       ROUND(AVG(confidence), 4) as avg_confidence
                FROM chat_history
                WHERE intent NOT IN ('default', 'empty', 'gibberish',
                                     'unclear_retype', 'error')
                  AND intent IS NOT NULL
                GROUP BY intent
                ORDER BY count DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_unresolved_queries_detailed(self, limit=30):
        """Get recent unresolved queries with details"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT uq.id, uq.user_message, uq.confidence,
                       uq.timestamp, uq.resolved, uq.admin_response,
                       uq.session_id
                FROM unresolved_queries uq
                WHERE uq.resolved = 0
                ORDER BY uq.timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_session_stats(self):
        """Get session-level statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Active sessions today
            cursor.execute('''
                SELECT COUNT(DISTINCT session_id) as today_sessions
                FROM chat_history
                WHERE DATE(timestamp) = DATE('now')
            ''')
            today_sessions = cursor.fetchone()['today_sessions']

            # Average chats per session
            cursor.execute('''
                SELECT ROUND(AVG(chat_count), 1) as avg_per_session
                FROM (
                    SELECT session_id, COUNT(*) as chat_count
                    FROM chat_history
                    GROUP BY session_id
                )
            ''')
            avg_per_session = cursor.fetchone()['avg_per_session'] or 0

            # Peak hour
            cursor.execute('''
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                       COUNT(*) as count
                FROM chat_history
                GROUP BY hour
                ORDER BY count DESC
                LIMIT 1
            ''')
            peak = cursor.fetchone()
            peak_hour = peak['hour'] if peak else 0

            # Resolve rate
            cursor.execute('SELECT COUNT(*) as total FROM unresolved_queries')
            total_unresolved_all = cursor.fetchone()['total']
            cursor.execute('SELECT COUNT(*) as resolved FROM unresolved_queries WHERE resolved = 1')
            resolved = cursor.fetchone()['resolved']
            resolve_rate = (resolved / total_unresolved_all * 100) if total_unresolved_all > 0 else 100

            return {
                'today_sessions': today_sessions,
                'avg_chats_per_session': avg_per_session,
                'peak_hour': peak_hour,
                'resolve_rate': round(resolve_rate, 1)
            }

    def get_weekly_comparison(self):
        """Compare this week vs last week"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) as count FROM chat_history
                WHERE timestamp >= datetime('now', '-7 days')
            ''')
            this_week = cursor.fetchone()['count']

            cursor.execute('''
                SELECT COUNT(*) as count FROM chat_history
                WHERE timestamp >= datetime('now', '-14 days')
                  AND timestamp < datetime('now', '-7 days')
            ''')
            last_week = cursor.fetchone()['count']

            change = 0
            if last_week > 0:
                change = round(((this_week - last_week) / last_week) * 100, 1)

            return {
                'this_week': this_week,
                'last_week': last_week,
                'change_percent': change
            }

    def get_recent_chats(self, limit=20):
        """Get recent chats for live view"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, session_id, user_message, bot_response,
                       intent, confidence, method, timestamp
                FROM chat_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # ══════════════════════════════════════
    # 📥 NEW: Export Data for CSV/PDF
    # ══════════════════════════════════════

    def get_export_data(self, limit=1000, date_from='', date_to='', intent_filter=''):
        """Get chat data for export with filters"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = 'SELECT * FROM chat_history WHERE 1=1'
            params = []

            if date_from:
                query += ' AND DATE(timestamp) >= ?'
                params.append(date_from)

            if date_to:
                query += ' AND DATE(timestamp) <= ?'
                params.append(date_to)

            if intent_filter:
                query += ' AND intent = ?'
                params.append(intent_filter)

            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
