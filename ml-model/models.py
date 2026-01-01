"""
SQLAlchemy models for NBA prediction database
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Index


db = SQLAlchemy()


class Prediction(db.Model):
    """Store all predictions and their outcomes"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    home_team = db.Column(db.String(50), nullable=False)
    away_team = db.Column(db.String(50), nullable=False)
    predicted_winner = db.Column(db.String(50), nullable=False)
    predicted_confidence = db.Column(db.Float, nullable=False)
    
    game_date = db.Column(db.DateTime, nullable=False, index=True)
    game_time_et = db.Column(db.String(20), nullable=True)
    
    # ✅ NEW FIELDS: SAVE #1 - When prediction is created
    created_by_user = db.Column(db.String(100), nullable=True)  # Track who made the prediction
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # When created
    
    # ✅ NEW FIELDS: SAVE #2 - When game completes
    actual_winner = db.Column(db.String(50), nullable=True)
    actual_score_home = db.Column(db.Integer, nullable=True)  # Final home score
    actual_score_away = db.Column(db.Integer, nullable=True)  # Final away score
    is_correct = db.Column(db.Boolean, nullable=True)
    result_updated_at = db.Column(db.DateTime, nullable=True)  # When result was verified
    
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    status = db.Column(db.String(20), default='pending')
    
    __table_args__ = (
        Index('idx_game_date', 'game_date'),
        Index('idx_status', 'status'),
        Index('idx_home_away', 'home_team', 'away_team'),
        Index('idx_created_by_user', 'created_by_user'),  # Index for user tracking
    )
    
    def to_dict(self):
        """Convert prediction to dictionary"""
        return {
            'id': self.id,
            'game_id': self.game_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'predicted_winner': self.predicted_winner,
            'predicted_confidence': round(self.predicted_confidence * 100, 2),
            'game_date': self.game_date.isoformat(),
            'game_time_et': self.game_time_et,
            'actual_winner': self.actual_winner,
            'actual_score': f"{self.actual_score_home}-{self.actual_score_away}" if self.actual_score_home else None,
            'is_correct': self.is_correct,
            'status': self.status,
            'created_by_user': self.created_by_user,  # ✅ NEW: Who made it
            'created_at': self.created_at.isoformat(),  # ✅ NEW: When made
            'predicted_at': self.predicted_at.isoformat(),
            'result_updated_at': self.result_updated_at.isoformat() if self.result_updated_at else None,  # ✅ NEW: When verified
        }



class GameSchedule(db.Model):
    """Store upcoming NBA games from external API"""
    __tablename__ = 'game_schedules'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    home_team = db.Column(db.String(50), nullable=False)
    away_team = db.Column(db.String(50), nullable=False)
    game_date = db.Column(db.DateTime, nullable=False, index=True)
    game_time_et = db.Column(db.String(20), nullable=True)
    arena = db.Column(db.String(100), nullable=True)
    season = db.Column(db.Integer, nullable=False)
    
    status = db.Column(db.String(20), default='scheduled')
    
    home_score = db.Column(db.Integer, nullable=True)
    away_score = db.Column(db.Integer, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_game_date_schedule', 'game_date'),
        Index('idx_status_schedule', 'status'),
    )
    
    def to_dict(self):
        return {
            'game_id': self.game_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'game_date': self.game_date.isoformat(),
            'game_time_et': self.game_time_et,
            'arena': self.arena,
            'status': self.status,
            'score': f"{self.home_score}-{self.away_score}" if self.home_score else None,
        }



class AccuracyMetrics(db.Model):
    """Cache accuracy metrics for quick retrieval"""
    __tablename__ = 'accuracy_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    
    total_predictions = db.Column(db.Integer, default=0)
    correct_predictions = db.Column(db.Integer, default=0)
    incorrect_predictions = db.Column(db.Integer, default=0)
    overall_accuracy = db.Column(db.Float, default=0.0)
    
    last_10_wins = db.Column(db.Integer, default=0)
    last_10_losses = db.Column(db.Integer, default=0)
    last_20_wins = db.Column(db.Integer, default=0)
    last_20_losses = db.Column(db.Integer, default=0)
    
    current_streak = db.Column(db.Integer, default=0)
    longest_win_streak = db.Column(db.Integer, default=0)
    longest_loss_streak = db.Column(db.Integer, default=0)
    
    best_team = db.Column(db.String(50), nullable=True)
    best_team_accuracy = db.Column(db.Float, nullable=True)
    worst_team = db.Column(db.String(50), nullable=True)
    worst_team_accuracy = db.Column(db.Float, nullable=True)
    
    calculated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'incorrect_predictions': self.incorrect_predictions,
            'overall_accuracy': round(self.overall_accuracy * 100, 2),
            'last_10': {
                'wins': self.last_10_wins,
                'losses': self.last_10_losses,
                'accuracy': round((self.last_10_wins / (self.last_10_wins + self.last_10_losses) * 100), 2) if (self.last_10_wins + self.last_10_losses) > 0 else 0
            },
            'last_20': {
                'wins': self.last_20_wins,
                'losses': self.last_20_losses,
                'accuracy': round((self.last_20_wins / (self.last_20_wins + self.last_20_losses) * 100), 2) if (self.last_20_wins + self.last_20_losses) > 0 else 0
            },
            'current_streak': self.current_streak,
            'longest_win_streak': self.longest_win_streak,
            'longest_loss_streak': self.longest_loss_streak,
            'best_team': self.best_team,
            'best_team_accuracy': round(self.best_team_accuracy * 100, 2) if self.best_team_accuracy else None,
            'worst_team': self.worst_team,
            'worst_team_accuracy': round(self.worst_team_accuracy * 100, 2) if self.worst_team_accuracy else None,
            'calculated_at': self.calculated_at.isoformat(),
        }