from datetime import timedelta
import re
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
import bcrypt
from ..config import settings
from ..utils.mongo import db, utcnow
bearer=HTTPBearer(auto_error=False)
def validate_password(password):
    if len(password.encode("utf-8")) > 72: raise HTTPException(422, "Password must not exceed 72 UTF-8 bytes.")
    if len(password)<8 or not re.search(r"[A-Z]",password) or not re.search(r"[a-z]",password) or not re.search(r"\d",password): raise HTTPException(422,"Password must be at least 8 characters and include uppercase, lowercase, and a number.")
def hash_password(password):
    validate_password(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
def verify_password(password,password_hash):
    try: return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError): return False
def _secret():
    secret = settings.JWT_SECRET or "visioninspect-dev-secret-change-me"
    return secret
def token_for(user): return jwt.encode({"sub":str(user["_id"]),"role":user["role"],"machine_id":user.get("machine_id"),"exp":utcnow()+timedelta(minutes=settings.JWT_EXPIRE_MINUTES)},_secret(),algorithm="HS256")
def current_user(credentials: HTTPAuthorizationCredentials|None=Depends(bearer)):
    if not credentials: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Authentication required.")
    try: payload=jwt.decode(credentials.credentials,_secret(),algorithms=["HS256"])
    except jwt.PyJWTError: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Invalid or expired token.")
    user=db().users.find_one({"_id":payload.get("sub"),"is_active":True})
    if not user: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Account is inactive or unavailable.")
    return user
def require_admin(user=Depends(current_user)):
    return user
