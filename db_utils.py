import sqlite3
import streamlit as st

class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            try:
                cls._instance.conn = sqlite3.connect('llm_cache.db', check_same_thread=False)
                cls._instance.cursor = cls._instance.conn.cursor()
                cls._instance.cursor.execute('''CREATE TABLE IF NOT EXISTS llm_cache
                                             (prompt TEXT PRIMARY KEY, response TEXT)''')
                cls._instance.conn.commit()
            except Exception as e:
                st.error(f"Failed to initialize SQLite database: {e}")
                st.stop()
        return cls._instance

    def get_connection(self):
        return self.conn, self.cursor

# Global instance
db_instance = Database()

def get_db_connection():
    return db_instance.get_connection()