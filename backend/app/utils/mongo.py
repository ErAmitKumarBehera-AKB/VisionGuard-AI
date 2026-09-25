"""MongoDB connection, indexes, and optional first-run account initialization."""
from datetime import datetime, timezone
import uuid
import bcrypt
from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError
from ..config import settings

_client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, socketTimeoutMS=5000, maxPoolSize=20, retryWrites=True)

def db():
    return _client[settings.MONGO_DB]

def utcnow():
    return datetime.now(timezone.utc)

def mongo_available():
    try:
        _client.admin.command("ping")
        return True
    except PyMongoError:
        return False

def _password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

def _seed_configured_accounts(database):
    machine_id = "machine-edge-04"
    database.machines.update_one(
        {"machine_id": machine_id},
        {"$setOnInsert": {"_id": uuid.uuid4().hex, "machine_id": machine_id, "machine_code": "EDGE-04", "name": "Northline Plant 04", "description": "Primary visual inspection line", "status": "ACTIVE", "created_at": utcnow(), "updated_at": utcnow()}},
        upsert=True,
    )
    admin_email = settings.ADMIN_EMAIL or settings.BOOTSTRAP_ADMIN_EMAIL
    admin_password = settings.ADMIN_PASSWORD or settings.BOOTSTRAP_ADMIN_PASSWORD
    accounts = [(admin_email, admin_password, "Workspace User", "USER", None, "user-001")]
    for email, password, name, role, assigned_machine, employee_id in accounts:
        if not email or not password:
            continue
        database.users.update_one(
            {"email": email.strip().lower()},
            {"$setOnInsert": {"_id": uuid.uuid4().hex, "full_name": name, "email": email.strip().lower(), "employee_id": employee_id, "password_hash": _password_hash(password), "role": role, "machine_id": assigned_machine, "is_active": True, "created_at": utcnow(), "updated_at": utcnow()}},
            upsert=True,
        )

def init_mongo():
    database = db()
    database.users.create_index([("email", ASCENDING)], unique=True)
    database.users.create_index([("employee_id", ASCENDING)], unique=True, sparse=True)
    database.users.create_index([("machine_id", ASCENDING)])
    database.machines.create_index([("machine_code", ASCENDING)], unique=True)
    database.inspections.create_index([("inspection_id", ASCENDING)], unique=True)
    database.inspections.create_index([("machine_id", ASCENDING), ("timestamp", ASCENDING)])
    database.inspections.create_index([("prediction", ASCENDING), ("confidence", ASCENDING)])
    database.inspections.create_index([("review_required", ASCENDING)])
    database.feedback.create_index([("inspection_id", ASCENDING)])
    database.feedback.create_index([("machine_id", ASCENDING), ("included_in_training", ASCENDING)])
    _seed_configured_accounts(database)
