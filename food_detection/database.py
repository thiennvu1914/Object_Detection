"""
Database Module
===============
SQLite database manager for caching embeddings and logging detection history.
"""
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class DatabaseManager:
    """SQLite database manager for food detection system"""
    
    def __init__(self, db_path: str = "food_detection.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Table 1: reference_embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reference_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_class_name 
            ON reference_embeddings(class_name)
        """)
        
        # Table 2: detection_sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                total_objects INTEGER NOT NULL,
                detected_classes TEXT NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detected_at 
            ON detection_sessions(detected_at)
        """)
        
        # Table 3: detected_objects
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detected_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                similarity REAL NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES detection_sessions(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON detected_objects(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_obj_class_name 
            ON detected_objects(class_name)
        """)
        
        self.conn.commit()
    
    # ==================== REFERENCE EMBEDDINGS ====================
    
    def save_reference_embeddings(self, class_name: str, image_path: str, embedding: np.ndarray):
        """
        Save a reference embedding to database.
        
        Args:
            class_name: Food class name
            image_path: Path to reference image
            embedding: 512-dim numpy array
        """
        cursor = self.conn.cursor()
        embedding_blob = embedding.tobytes()
        
        cursor.execute("""
            INSERT INTO reference_embeddings (class_name, image_path, embedding)
            VALUES (?, ?, ?)
        """, (class_name, image_path, embedding_blob))
        
        self.conn.commit()
    
    def load_reference_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load all reference embeddings from database.
        
        Returns:
            Dictionary mapping class_name to embeddings array (N x 512)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT class_name, embedding 
            FROM reference_embeddings
            ORDER BY class_name, id
        """)
        
        ref_embeddings = {}
        for row in cursor.fetchall():
            class_name = row['class_name']
            embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            
            if class_name not in ref_embeddings:
                ref_embeddings[class_name] = []
            
            ref_embeddings[class_name].append(embedding)
        
        # Convert lists to numpy arrays
        for class_name in ref_embeddings:
            ref_embeddings[class_name] = np.array(ref_embeddings[class_name])
        
        return ref_embeddings
    
    def clear_reference_embeddings(self):
        """Delete all reference embeddings (for re-caching)"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM reference_embeddings")
        self.conn.commit()
    
    def get_embeddings_count(self) -> Dict[str, int]:
        """
        Get count of reference embeddings per class.
        
        Returns:
            Dictionary mapping class_name to count
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT class_name, COUNT(*) as count
            FROM reference_embeddings
            GROUP BY class_name
            ORDER BY class_name
        """)
        
        return {row['class_name']: row['count'] for row in cursor.fetchall()}
    
    # ==================== DETECTION HISTORY ====================
    
    def save_detection_session(
        self, 
        image_filename: str, 
        detections: List[Dict]
    ) -> int:
        """
        Save a detection session with all detected objects.
        
        Args:
            image_filename: Name of uploaded image
            detections: List of detection dicts with keys: class, similarity, confidence
            
        Returns:
            Session ID
        """
        cursor = self.conn.cursor()
        
        # Extract data
        total_objects = len(detections)
        detected_classes = [det['class'] for det in detections]
        detected_classes_json = json.dumps(detected_classes)
        
        # Insert session
        cursor.execute("""
            INSERT INTO detection_sessions (image_filename, total_objects, detected_classes)
            VALUES (?, ?, ?)
        """, (image_filename, total_objects, detected_classes_json))
        
        session_id = cursor.lastrowid
        
        # Insert detected objects
        for det in detections:
            cursor.execute("""
                INSERT INTO detected_objects (session_id, class_name, similarity, confidence)
                VALUES (?, ?, ?, ?)
            """, (session_id, det['class'], det['similarity'], det['confidence']))
        
        self.conn.commit()
        return session_id
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent detection sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dicts
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, image_filename, total_objects, detected_classes, detected_at
            FROM detection_sessions
            ORDER BY detected_at DESC
            LIMIT ?
        """, (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row['id'],
                'image_filename': row['image_filename'],
                'total_objects': row['total_objects'],
                'detected_classes': json.loads(row['detected_classes']),
                'detected_at': row['detected_at']
            })
        
        return sessions
    
    def get_session_details(self, session_id: int) -> Optional[Dict]:
        """
        Get detailed information about a detection session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session dict with objects list, or None if not found
        """
        cursor = self.conn.cursor()
        
        # Get session info
        cursor.execute("""
            SELECT id, image_filename, total_objects, detected_classes, detected_at
            FROM detection_sessions
            WHERE id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        session = {
            'id': row['id'],
            'image_filename': row['image_filename'],
            'total_objects': row['total_objects'],
            'detected_classes': json.loads(row['detected_classes']),
            'detected_at': row['detected_at']
        }
        
        # Get detected objects
        cursor.execute("""
            SELECT class_name, similarity, confidence
            FROM detected_objects
            WHERE session_id = ?
            ORDER BY similarity DESC
        """, (session_id,))
        
        session['objects'] = [
            {
                'class': row['class_name'],
                'similarity': row['similarity'],
                'confidence': row['confidence']
            }
            for row in cursor.fetchall()
        ]
        
        return session
    
    def get_class_statistics(self) -> List[Dict]:
        """
        Get detection statistics per class.
        
        Returns:
            List of dicts with class_name, total_detections, avg_similarity
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                class_name,
                COUNT(*) as total_detections,
                AVG(similarity) as avg_similarity,
                MAX(similarity) as max_similarity,
                MIN(similarity) as min_similarity
            FROM detected_objects
            GROUP BY class_name
            ORDER BY total_detections DESC
        """)
        
        return [
            {
                'class_name': row['class_name'],
                'total_detections': row['total_detections'],
                'avg_similarity': round(row['avg_similarity'], 3),
                'max_similarity': round(row['max_similarity'], 3),
                'min_similarity': round(row['min_similarity'], 3)
            }
            for row in cursor.fetchall()
        ]
    
    # ==================== UTILITY ====================
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
