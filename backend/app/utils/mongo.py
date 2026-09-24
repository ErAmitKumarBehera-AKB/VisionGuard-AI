"""MongoDB helpers for the authenticated API."""
from datetime import datetime, timezone
from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError
from ..config import settings
_client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000)
def db(): return _client[settings.MONGO_DB]
def utcnow(): return datetime.now(timezone.utc)
def init_mongo():
    d=db()
    d.users.create_index([("email", ASCENDING)], unique=True); d.users.create_index([("employee_id", ASCENDING)], unique=True, sparse=True); d.users.create_index([("machine_id", ASCENDING)])
    d.machines.create_index([("machine_code", ASCENDING)], unique=True)
    d.inspections.create_index([("inspection_id", ASCENDING)], unique=True); d.inspections.create_index([("machine_id", ASCENDING), ("timestamp", ASCENDING)]); d.inspections.create_index([("prediction", ASCENDING), ("confidence", ASCENDING)]); d.inspections.create_index([("review_required", ASCENDING)])
    d.feedback.create_index([("inspection_id", ASCENDING)]); d.feedback.create_index([("machine_id", ASCENDING), ("included_in_training", ASCENDING)])
def mongo_available():
    try: _client.admin.command("ping"); return True
    except PyMongoError: return False
