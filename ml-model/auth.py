from functools import wraps
from datetime import datetime, timedelta
import jwt
import os
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from flask import request, jsonify

# ==================== LOGGING CONFIGURATION ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
TOKEN_EXPIRATION_DAYS = int(os.getenv('TOKEN_EXPIRATION_DAYS', 30))

# ==================== IN-MEMORY USER STORE ====================
# For production, use a database (PostgreSQL, MongoDB, etc.)

users_db = {}

# ==================== USER CLASS ====================

class User:
    """User model for authentication"""
    
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = generate_password_hash(password)
        self.created_at = datetime.now()
        self.last_login = None
        self.is_active = True
    
    def verify_password(self, password):
        """Verify password against hash"""
        return check_password_hash(self.password, password)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.now()
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    def to_dict_safe(self):
        """Convert user to safe dictionary (no sensitive info)"""
        return {
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }

# ==================== JWT TOKEN MANAGEMENT ====================

def generate_token(username):
    """
    Generate JWT token for user
    
    Args:
        username (str): Username to encode in token
    
    Returns:
        str: Encoded JWT token
    """
    try:
        payload = {
            'username': username,
            'exp': datetime.utcnow() + timedelta(days=TOKEN_EXPIRATION_DAYS),
            'iat': datetime.utcnow()
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        logger.info(f"✓ Token generated for user: {username}")
        return token
    except Exception as e:
        logger.error(f"Error generating token: {e}")
        raise

def verify_token(token):
    """
    Verify JWT token
    
    Args:
        token (str): JWT token to verify
    
    Returns:
        dict: Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        return None

# ==================== DECORATOR FOR PROTECTED ROUTES ====================

def token_required(f):
    """
    Decorator to protect routes that require authentication
    
    Usage:
        @app.route('/protected')
        @token_required
        def protected_route():
            username = request.user['username']
            return jsonify({'user': username})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Extract token from Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                # Expected format: "Bearer <token>"
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'error': 'Invalid token format. Use: Authorization: Bearer <token>'}), 401
        
        # Token is missing
        if not token:
            return jsonify({'error': 'Token is missing. Please provide a valid token in Authorization header.'}), 401
        
        # Verify token
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Attach user info to request object
        request.user = payload
        return f(*args, **kwargs)
    
    return decorated

# ==================== AUTH HANDLER FUNCTIONS ====================

def handle_register(data):
    """
    Handle user registration
    
    Args:
        data (dict): Request data containing username, email, password
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        # Validate required fields
        if not data:
            return {'error': 'Request body is empty'}, 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not username or not email or not password:
            return {
                'error': 'Missing required fields',
                'required_fields': ['username', 'email', 'password']
            }, 400
        
        # Validate username
        if len(username) < 3:
            return {'error': 'Username must be at least 3 characters long'}, 400
        
        if len(username) > 50:
            return {'error': 'Username must be at most 50 characters long'}, 400
        
        # Validate email
        if '@' not in email or '.' not in email:
            return {'error': 'Invalid email format'}, 400
        
        if len(email) > 120:
            return {'error': 'Email must be at most 120 characters long'}, 400
        
        # Validate password
        if len(password) < 6:
            return {'error': 'Password must be at least 6 characters long'}, 400
        
        if len(password) > 120:
            return {'error': 'Password must be at most 120 characters long'}, 400
        
        # Check if username already exists
        if username in users_db:
            logger.warning(f"Registration failed: Username '{username}' already exists")
            return {'error': 'Username already exists'}, 409
        
        # Check if email already exists
        for user in users_db.values():
            if user.email == email:
                logger.warning(f"Registration failed: Email '{email}' already exists")
                return {'error': 'Email already exists'}, 409
        
        # Create new user
        user = User(username, email, password)
        users_db[username] = user
        
        # Generate token
        token = generate_token(username)
        
        logger.info(f"✓ New user registered: {username} ({email})")
        
        return {
            'success': True,
            'message': 'User registered successfully',
            'user': user.to_dict_safe(),
            'token': token
        }, 201
    
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return {'error': str(e)}, 500

def handle_login(data):
    """
    Handle user login
    
    Args:
        data (dict): Request data containing username, password
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        # Validate required fields
        if not data:
            return {'error': 'Request body is empty'}, 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return {
                'error': 'Missing credentials',
                'required_fields': ['username', 'password']
            }, 400
        
        # Check if user exists
        if username not in users_db:
            logger.warning(f"Login failed: User '{username}' not found")
            return {'error': 'Invalid username or password'}, 401
        
        user = users_db[username]
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Login failed: User '{username}' is inactive")
            return {'error': 'User account is inactive'}, 403
        
        # Verify password
        if not user.verify_password(password):
            logger.warning(f"Login failed: Invalid password for user '{username}'")
            return {'error': 'Invalid username or password'}, 401
        
        # Update last login
        user.update_last_login()
        
        # Generate token
        token = generate_token(username)
        
        logger.info(f"✓ User logged in: {username}")
        
        return {
            'success': True,
            'message': 'Login successful',
            'user': user.to_dict_safe(),
            'token': token
        }, 200
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return {'error': str(e)}, 500

def handle_get_current_user(username):
    """
    Get current logged-in user information
    
    Args:
        username (str): Username from JWT token
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        if username not in users_db:
            logger.warning(f"User not found: {username}")
            return {'error': 'User not found'}, 404
        
        user = users_db[username]
        
        return {
            'success': True,
            'user': user.to_dict_safe()
        }, 200
    
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return {'error': str(e)}, 500

def handle_logout(username):
    """
    Handle user logout (optional - token just expires)
    
    Args:
        username (str): Username from JWT token
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        logger.info(f"✓ User logged out: {username}")
        return {
            'success': True,
            'message': 'Logout successful'
        }, 200
    
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {'error': str(e)}, 500

def handle_change_password(username, data):
    """
    Handle password change
    
    Args:
        username (str): Username from JWT token
        data (dict): Request data containing old_password, new_password
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        # Validate required fields
        if not data:
            return {'error': 'Request body is empty'}, 400
        
        old_password = data.get('old_password', '')
        new_password = data.get('new_password', '')
        
        if not old_password or not new_password:
            return {
                'error': 'Missing required fields',
                'required_fields': ['old_password', 'new_password']
            }, 400
        
        # Check if user exists
        if username not in users_db:
            return {'error': 'User not found'}, 404
        
        user = users_db[username]
        
        # Verify old password
        if not user.verify_password(old_password):
            logger.warning(f"Password change failed: Invalid old password for {username}")
            return {'error': 'Invalid old password'}, 401
        
        # Validate new password
        if len(new_password) < 6:
            return {'error': 'New password must be at least 6 characters long'}, 400
        
        if len(new_password) > 120:
            return {'error': 'New password must be at most 120 characters long'}, 400
        
        # Check if new password is same as old password
        if user.verify_password(new_password):
            return {'error': 'New password cannot be the same as old password'}, 400
        
        # Update password
        user.password = generate_password_hash(new_password)
        
        logger.info(f"✓ Password changed for user: {username}")
        
        return {
            'success': True,
            'message': 'Password changed successfully'
        }, 200
    
    except Exception as e:
        logger.error(f"Password change error: {e}")
        return {'error': str(e)}, 500

def handle_delete_account(username, data):
    """
    Handle account deletion
    
    Args:
        username (str): Username from JWT token
        data (dict): Request data containing password for confirmation
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        # Validate required fields
        if not data:
            return {'error': 'Request body is empty'}, 400
        
        password = data.get('password', '')
        
        if not password:
            return {'error': 'Password required for account deletion'}, 400
        
        # Check if user exists
        if username not in users_db:
            return {'error': 'User not found'}, 404
        
        user = users_db[username]
        
        # Verify password
        if not user.verify_password(password):
            logger.warning(f"Account deletion failed: Invalid password for {username}")
            return {'error': 'Invalid password'}, 401
        
        # Delete user
        del users_db[username]
        
        logger.info(f"✓ Account deleted for user: {username}")
        
        return {
            'success': True,
            'message': 'Account deleted successfully'
        }, 200
    
    except Exception as e:
        logger.error(f"Account deletion error: {e}")
        return {'error': str(e)}, 500

# ==================== UTILITY FUNCTIONS ====================

def get_all_users():
    """
    Get all users (admin function - for debugging only)
    
    Returns:
        dict: User information
    """
    return {
        'total_users': len(users_db),
        'users': {username: user.to_dict_safe() for username, user in users_db.items()}
    }

def user_exists(username):
    """
    Check if user exists
    
    Args:
        username (str): Username to check
    
    Returns:
        bool: True if user exists, False otherwise
    """
    return username in users_db