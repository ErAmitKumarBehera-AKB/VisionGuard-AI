"""Authenticated MongoDB API: roles, machines, inspections, feedback, and retraining."""
from datetime import datetime
from pathlib import Path
import subprocess, uuid
from io import BytesIO
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pymongo.errors import DuplicateKeyError
from ...auth.security import current_user, hash_password, require_admin, token_for, verify_password
from ...config import settings
from ...services.bentoml_client import BentoMLClient
from ...services.decision_service import decide
from ...utils.mongo import db, utcnow, mongo_available

router=APIRouter(tags=["Authenticated inspection workflow"])
bento=BentoMLClient()
ALLOWED={"image/jpeg":".jpg","image/png":".png","image/webp":".webp"}

def clean(doc):
    if not doc: return None
    doc=dict(doc); doc.pop("_id",None)
    if doc.get("inspection_id") and doc.get("image_reference"):
        doc["image_url"] = f"/api/v1/inspection-evidence/{doc['inspection_id']}"
    for k,v in doc.items():
        if isinstance(v,datetime): doc[k]=v.isoformat()
    return doc

def audit(actor, action, target=None, details=None):
    db().audit_logs.insert_one({"actor_id":actor.get("_id") if actor else None,"action":action,"target":target,"details":details or {},"timestamp":utcnow()})

def ensure_mongo():
    if not mongo_available(): raise HTTPException(503,"MongoDB is unavailable.")

def inspection_id(): return f"INS-{utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


def user_view(u):
    view = {k:v for k,v in clean(u).items() if k not in {"password_hash"}}
    view["role"] = "USER"
    return view

@router.post("/auth/register", status_code=201, summary="Create a workspace user account")
def register_user(payload:dict):
    ensure_mongo()
    email=str(payload.get("email", "")).strip().lower()
    password=str(payload.get("password", ""))
    full_name=str(payload.get("full_name", "")).strip()
    employee_id=str(payload.get("employee_id", "")).strip()
    if not full_name or not employee_id: raise HTTPException(422, "Full name and employee ID are required.")
    if not email or "@" not in email: raise HTTPException(422, "A valid email address is required.")
    doc={"_id":uuid.uuid4().hex,"full_name":full_name,"email":email,"employee_id":employee_id,"password_hash":hash_password(password),"role":"USER","machine_id":None,"is_active":True,"created_at":utcnow(),"updated_at":utcnow()}
    try: db().users.insert_one(doc)
    except DuplicateKeyError: raise HTTPException(409, "An account with this email or employee ID already exists.")
    audit(doc,"USER_CREATED",doc["_id"])
    return user_view(doc)

@router.post("/auth/login", summary="Login using an administrator-created account")
def login(payload:dict):
    ensure_mongo(); email=str(payload.get("email","")).strip().lower(); password=str(payload.get("password", ""))
    user=db().users.find_one({"email":email})
    if not user or not user.get("is_active") or not verify_password(password,user["password_hash"]): raise HTTPException(401,"Invalid email or password.")
    db().users.update_one({"_id":user["_id"]},{"$set":{"last_login":utcnow()}}); audit(user,"LOGIN")
    return {"access_token":token_for(user),"token_type":"bearer","user":user_view(user)}

@router.get("/auth/me")
def me(user=Depends(require_admin)): return user_view(user)

@router.put("/auth/me/password")
def change_password(payload:dict, user=Depends(require_admin)):
    current=str(payload.get("current_password", ""))
    new=str(payload.get("new_password", ""))
    if not verify_password(current, user.get("password_hash", "")):
        raise HTTPException(401, "Current password is incorrect.")
    hashed=hash_password(new)
    db().users.update_one({"_id":user["_id"]},{"$set":{"password_hash":hashed,"updated_at":utcnow()}})
    audit(user, "PASSWORD_CHANGED", user["_id"])
    return {"status":"password_changed"}

@router.put("/auth/me")
def update_me(payload:dict, user=Depends(require_admin)):
    allowed = {key: str(payload[key]).strip() for key in ("full_name", "organization", "job_title", "employee_id", "plant_name") if key in payload}
    if any(not value for value in allowed.values()):
        raise HTTPException(422, "Profile fields cannot be empty.")
    if "employee_id" in allowed and db().users.find_one({"employee_id": allowed["employee_id"], "_id": {"$ne": user["_id"]}}):
        raise HTTPException(409, "Employee ID already exists.")
    allowed["updated_at"] = utcnow()
    updated = db().users.find_one_and_update({"_id": user["_id"]}, {"$set": allowed}, return_document=True)
    audit(updated, "PROFILE_UPDATED", updated["_id"], {key: value for key, value in allowed.items() if key != "updated_at"})
    return user_view(updated)

@router.get("/admin/dashboard")
def admin_dashboard(admin=Depends(require_admin)):
    ensure_mongo(); d=db(); total=d.inspections.count_documents({"registered":True}); defects=d.inspections.count_documents({"registered":True,"final_label":"DEFECT"}); pending=d.inspections.count_documents({"prediction":"DEFECT","review_required":True,"review_completed":False})
    current_model=d.model_versions.find_one({"status":"PRODUCTION"},{"_id":0})
    checkpoint=Path(settings.MODEL_CHECKPOINT_PATH)
    if not current_model and checkpoint.is_file(): current_model={"model_name":"ResNet-50 Defect Detector","model_version":"checkpoint-"+checkpoint.stem,"architecture":"resnet50","status":"AVAILABLE","deployment_status":"LOCAL_CHECKPOINT"}
    return {"machine":clean(d.machines.find_one({"status":"ACTIVE"}, sort=[("machine_code",1)])),"total_inspections":total,"total_ok":d.inspections.count_documents({"registered":True,"final_label":"OK"}),"total_defect":defects,"defect_rate":round(defects/total*100,2) if total else 0,"pending_human_reviews":pending,"active_users":d.users.count_documents({"is_active":True}),"roles":1,"audit_events":d.audit_logs.count_documents({}),"policy_alerts":0,"administrators":d.users.count_documents({"role":"ADMIN","is_active":True}),"active_machines":d.machines.count_documents({"status":"ACTIVE"}),"current_model":current_model}

@router.get("/admin/inspections")
def all_inspections(machine_id: str | None = None, admin=Depends(require_admin)):
    ensure_mongo()
    query = {"prediction": "DEFECT"}
    if machine_id:
        query["machine_id"] = machine_id
    return [clean(item) for item in db().inspections.find(query).sort("timestamp", -1).limit(500)]

@router.get("/admin/inspections/{inspection_id}")
def admin_inspection(inspection_id: str, admin=Depends(require_admin)):
    ensure_mongo()
    record = db().inspections.find_one({"inspection_id": inspection_id})
    if not record:
        raise HTTPException(404, "Inspection not found.")
    return clean(record)

@router.get("/inspection-evidence/{inspection_id}")
def inspection_evidence(inspection_id: str, admin=Depends(require_admin)):
    ensure_mongo()
    record = db().inspections.find_one({"inspection_id": inspection_id})
    if not record:
        raise HTTPException(404, "Inspection not found.")
    reference = Path(str(record.get("image_reference", "")))
    if not reference.is_file():
        raise HTTPException(404, "Captured inspection image is unavailable.")
    media_type = {".png": "image/png", ".webp": "image/webp"}.get(reference.suffix.lower(), "image/jpeg")
    return FileResponse(reference, media_type=media_type, filename=reference.name)

@router.get("/admin/machines")
def machines(admin=Depends(require_admin)): ensure_mongo(); return [clean(x) for x in db().machines.find().sort("machine_code",1)]
@router.post("/admin/machines",status_code=201)
def create_machine(payload:dict,admin=Depends(require_admin)):
    ensure_mongo(); code=str(payload.get("machine_code","")).strip().upper(); name=str(payload.get("name","")).strip()
    if not code or not name: raise HTTPException(422,"Machine code and name are required.")
    machine_status=str(payload.get("status","ACTIVE")).upper()
    if machine_status not in {"ACTIVE", "INACTIVE", "MAINTENANCE"}: raise HTTPException(422,"Invalid machine status.")
    doc={"_id":uuid.uuid4().hex,"machine_id":uuid.uuid4().hex,"machine_code":code,"name":name,"description":str(payload.get("description", "")),"status":machine_status,"created_at":utcnow(),"updated_at":utcnow()}
    try: db().machines.insert_one(doc)
    except DuplicateKeyError: raise HTTPException(409,"Machine code already exists.")
    audit(admin,"MACHINE_CREATED",doc["machine_id"]); return clean(doc)
@router.put("/admin/machines/{machine_id}")
def update_machine(machine_id:str,payload:dict,admin=Depends(require_admin)):
    ensure_mongo(); changes={k:payload[k] for k in ("name","description","status","machine_code") if k in payload}
    if "status" in changes and str(changes["status"]).upper() not in {"ACTIVE", "INACTIVE", "MAINTENANCE"}: raise HTTPException(422,"Invalid machine status.")
    if "status" in changes: changes["status"]=str(changes["status"]).upper()
    changes["updated_at"]=utcnow()
    rec=db().machines.find_one_and_update({"machine_id":machine_id},{"$set":changes},return_document=True)
    if not rec: raise HTTPException(404,"Machine not found.")
    audit(admin,"MACHINE_UPDATED",machine_id,changes); return clean(rec)

@router.post("/admin/inspect",status_code=201)
async def inspect(image:UploadFile=File(...), product_category:str|None=Form(None), user=Depends(require_admin)):
    ensure_mongo()
    machine_id = user.get("machine_id")
    if not machine_id:
        machine = db().machines.find_one({"status": "ACTIVE"}, sort=[("machine_code", 1)])
        machine_id = machine.get("machine_id") if machine else None
    if not machine_id:
        raise HTTPException(422, "No active inspection machine is configured.")
    if image.content_type not in ALLOWED: raise HTTPException(415,"Only JPEG, PNG, and WEBP images are accepted.")
    body=await image.read(settings.MAX_UPLOAD_BYTES + 1)
    if not body: raise HTTPException(400,"Image is empty.")
    if len(body)>settings.MAX_UPLOAD_BYTES: raise HTTPException(413,"Image exceeds the configured 10 MB limit.")
    try:
        with Image.open(BytesIO(body)) as decoded: decoded.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The uploaded file is not a valid image.")
    iid=inspection_id(); storage=Path(settings.IMAGE_STORAGE_PATH); storage.mkdir(parents=True,exist_ok=True); reference=storage/f"{iid}.jpg"; reference.write_bytes(body)
    try: result=await bento.predict(body,filename=f"{iid}{ALLOWED[image.content_type]}",product_category=product_category)
    except Exception: raise HTTPException(503,"Model service unavailable. Try again.")
    prediction=str(result.get("prediction","")).upper(); confidence=float(result.get("confidence",0));
    if prediction not in {"OK","DEFECT"} or not 0<=confidence<=1: raise HTTPException(503,"Model service returned an invalid prediction.")
    # This portal registers confirmed defects only. Healthy frames are discarded after inference.
    if prediction == "OK":
        reference.unlink(missing_ok=True)
        audit(user, "OK_FRAME_IGNORED", iid, {"confidence": confidence})
        return {"inspection_id": iid, "timestamp": utcnow(), "prediction": "OK", "predicted_label": "OK", "confidence": confidence, "model_version": result.get("model_version", "unknown"), "inference_latency_ms": float(result.get("latency_ms", 0)), "status": "IGNORED_OK", "review_required": False, "review_completed": False, "registered": False, "image_url": None}
    doc={"_id":uuid.uuid4().hex,"inspection_id":iid,"machine_id":machine_id,"created_by":user["_id"],"timestamp":utcnow(),"image_reference":str(reference),"prediction":"DEFECT","predicted_label":"DEFECT","confidence":confidence,"model_name":result.get("model_name", "ResNet-50"),"model_version":result.get("model_version","unknown"),"inference_latency_ms":float(result.get("latency_ms",0)),"status":"PENDING_REVIEW","decision":"PENDING_HUMAN_REVIEW","final_label":None,"review_required":True,"review_completed":False,"registered":False,"operator_flagged":False,"operator_action":None,"operator_remark":None}
    db().inspections.insert_one(doc); audit(user,"DEFECT_CANDIDATE_CREATED",iid,{"prediction":"DEFECT","confidence":confidence}); return clean(doc)

@router.post("/admin/inspections/{inspection_id}/flag")
def flag(inspection_id:str,user=Depends(require_admin)):
    ensure_mongo(); rec=db().inspections.find_one_and_update({"inspection_id":inspection_id,"review_completed":False},{"$set":{"review_required":True,"operator_flagged":True,"status":"FLAGGED","decision":"PENDING_HUMAN_REVIEW","final_label":None}},return_document=True)
    if not rec: raise HTTPException(404,"Open inspection not found.")
    audit(user,"INSPECTION_FLAGGED",inspection_id); return clean(rec)
@router.post("/admin/inspections/{inspection_id}/review")
def review(inspection_id:str,payload:dict,user=Depends(require_admin)):
    ensure_mongo(); label=str(payload.get("human_label","")).upper(); remark=str(payload.get("remark","")).strip()
    if label not in {"OK","DEFECT"} or not remark: raise HTTPException(422,"A final label and non-empty remark are required.")
    query={"inspection_id": inspection_id}
    rec=db().inspections.find_one(query)
    if not rec: raise HTTPException(404,"Inspection not found.")
    if rec.get("review_completed"): raise HTTPException(409,"This inspection has already been reviewed.")
    # High-confidence DEFECT approval/denial and every pending/flagged review converge here.
    final_status="HUMAN_CONFIRMED_"+label
    feedback_id=uuid.uuid4().hex
    reviewed_at=utcnow()
    if label == "OK":
        # A denied defect is a false positive: do not retain it as a registered case.
        reference = Path(str(rec.get("image_reference", "")))
        reference.unlink(missing_ok=True)
        db().inspections.delete_one({"_id": rec["_id"]})
        audit(user, "DEFECT_CANDIDATE_DENIED", inspection_id, {"human_label": "OK", "remark": remark})
        return {"inspection_id": inspection_id, "status": "DISCARDED", "registered": False, "review_completed": True, "human_label": "OK"}
    db().inspections.update_one({"_id":rec["_id"]},{"$set":{"final_label":"DEFECT","human_label":"DEFECT","operator_action":"DEFECT","operator_remark":remark,"review_required":True,"review_completed":True,"registered":True,"reviewed_by":user["_id"],"reviewed_at":reviewed_at,"feedback_id":feedback_id,"status":"HUMAN_CONFIRMED_DEFECT","decision":"HUMAN_APPROVED"}})
    feedback={"_id":feedback_id,"feedback_id":feedback_id,"inspection_id":inspection_id,"image_reference":rec["image_reference"],"machine_id":rec["machine_id"],"created_by":rec["created_by"],"predicted_label":rec["predicted_label"],"human_label":"DEFECT","confidence":rec["confidence"],"remark":remark,"model_version":rec["model_version"],"created_at":reviewed_at,"included_in_training":False,"training_approved":False,"dataset_version":None}
    db().feedback.insert_one(feedback); audit(user,"DEFECT_APPROVED",inspection_id,{"human_label":"DEFECT","remark":remark}); return clean(db().inspections.find_one({"_id":rec["_id"]}))

@router.get("/feedback/pending")
def pending_feedback(user=Depends(require_admin)):
    ensure_mongo(); q={"prediction":"DEFECT","review_required":True,"review_completed":False};
    return [clean(x) for x in db().inspections.find(q).sort("timestamp",1)]

@router.get("/admin/feedback")
def feedback_pool(admin=Depends(require_admin)): ensure_mongo(); return [clean(x) for x in db().feedback.find().sort("created_at",-1)]
@router.post("/admin/feedback/{feedback_id}/approve")
def approve_feedback(feedback_id:str,payload:dict|None=None,admin=Depends(require_admin)):
    ensure_mongo(); rec=db().feedback.find_one_and_update({"feedback_id":feedback_id},{"$set":{"included_in_training":True,"training_approved":True,"dataset_version":(payload or {}).get("dataset_version")}},return_document=True)
    if not rec: raise HTTPException(404,"Feedback not found.")
    audit(admin,"FEEDBACK_APPROVED_FOR_TRAINING",feedback_id); return clean(rec)

@router.post("/admin/feedback/{feedback_id}/reject")
def reject_feedback(feedback_id:str, admin=Depends(require_admin)):
    ensure_mongo(); rec=db().feedback.find_one_and_update({"feedback_id":feedback_id},{"$set":{"included_in_training":False,"training_approved":False,"rejected_at":utcnow(),"rejected_by":admin["_id"]}},return_document=True)
    if not rec: raise HTTPException(404,"Feedback not found.")
    audit(admin,"FEEDBACK_REJECTED_FROM_TRAINING",feedback_id); return clean(rec)

@router.get("/models")
def models(user=Depends(require_admin)):
    ensure_mongo()
    registered = [clean(x) for x in db().model_versions.find().sort("created_at", -1)]
    if registered:
        return registered
    checkpoint = Path(settings.MODEL_CHECKPOINT_PATH)
    metadata_path = checkpoint.with_suffix(".json")
    metadata = {}
    if metadata_path.is_file():
        try:
            import json
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
    model_config = metadata.get("config", {}).get("model", {})
    metrics = metadata.get("metrics", {})
    if checkpoint.is_file():
        return [{
            "model_name": "ResNet-50 Defect Detector",
            "model_version": "checkpoint-" + checkpoint.stem,
            "architecture": model_config.get("architecture", "resnet50"),
            "status": "AVAILABLE",
            "deployment_status": "LOCAL_CHECKPOINT",
            "dataset_version": metadata.get("config", {}).get("mlflow", {}).get("registered_model_name", ""),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("defect_precision"),
            "recall": metrics.get("defect_recall"),
            "f1_score": metrics.get("defect_f1"),
            "created_at": None,
            "checkpoint_path": str(checkpoint),
        }]
    return []

@router.get("/admin/models")
def admin_models(admin=Depends(require_admin)):
    return models(user=admin)

@router.get("/notifications")
def notifications(user=Depends(require_admin)):
    ensure_mongo()
    logs = db().audit_logs.find({"actor_id": user["_id"]}).sort("timestamp", -1).limit(25)
    return [{"id": str(item.get("_id", item.get("timestamp"))), "type": "info", "title": item.get("action", "Workspace event").replace("_", " ").title(), "message": "Workspace activity recorded.", "timestamp": item.get("timestamp").isoformat() if item.get("timestamp") else utcnow().isoformat(), "read": False} for item in logs]

@router.post("/notifications/read")
def mark_notifications_read(user=Depends(require_admin)):
    return {"status": "ok"}

@router.get("/models/active")
def active_model(user=Depends(require_admin)):
    ensure_mongo(); active = clean(db().model_versions.find_one({"status":"PRODUCTION"}))
    if active:
        return active
    checkpoint = Path(settings.MODEL_CHECKPOINT_PATH)
    return {"status": "AVAILABLE", "model_name": "ResNet-50 Defect Detector", "model_version": "checkpoint-" + checkpoint.stem, "architecture": "resnet50", "checkpoint_available": checkpoint.is_file()}
@router.get("/retraining/status")
def retraining_status(admin=Depends(require_admin)):
    ensure_mongo(); latest=db().retraining_jobs.find_one(sort=[("started_at",-1)]); return clean(latest) or {"status":"IDLE","feedback_count":db().feedback.count_documents({"included_in_training":True})}

@router.get("/admin/retraining/status")
def admin_retraining_status(admin=Depends(require_admin)):
    return retraining_status(admin=admin)
@router.get("/retraining/jobs/{job_id}")
def retraining_job(job_id:str,admin=Depends(require_admin)):
    ensure_mongo(); job=db().retraining_jobs.find_one({"job_id":job_id})
    if not job: raise HTTPException(404,"Retraining job not found.")
    return clean(job)
@router.post("/retraining/trigger",status_code=202)
def trigger_retraining(admin=Depends(require_admin)):
    ensure_mongo(); job_id=uuid.uuid4().hex; approved=db().feedback.count_documents({"included_in_training":True}); job={"_id":uuid.uuid4().hex,"job_id":job_id,"status":"TRAINING_STARTED","progress":0,"feedback_count":approved,"dataset_version":None,"candidate_model_version":None,"metrics":None,"safety_gate":None,"started_at":utcnow(),"completed_at":None}
    db().retraining_jobs.insert_one(job); audit(admin,"RETRAINING_TRIGGERED",job_id,{"approved_feedback":approved})
    # Training is intentionally asynchronous; no fabricated metrics or automatic promotion.
    return {"status":"TRAINING_STARTED","job_id":job_id,"feedback_count":approved}

@router.post("/admin/retraining/trigger", status_code=202)
def admin_trigger_retraining(admin=Depends(require_admin)):
    return trigger_retraining(admin=admin)

@router.get("/admin/audit-logs")
def audit_logs(admin=Depends(require_admin), limit:int=Query(500, ge=1, le=2000)):
    ensure_mongo(); return [clean(x) for x in db().audit_logs.find().sort("timestamp", -1).limit(limit)]
