"""Authenticated MongoDB API: roles, machines, inspections, feedback, and retraining."""
from datetime import datetime
from pathlib import Path
import subprocess, uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pymongo.errors import DuplicateKeyError
from ...auth.security import current_user, hash_password, require_admin, require_supervisor, token_for, verify_password
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
    for k,v in doc.items():
        if isinstance(v,datetime): doc[k]=v.isoformat()
    return doc

def audit(actor, action, target=None, details=None):
    db().audit_logs.insert_one({"actor_id":actor.get("_id") if actor else None,"action":action,"target":target,"details":details or {},"timestamp":utcnow()})

def ensure_mongo():
    if not mongo_available(): raise HTTPException(503,"MongoDB is unavailable.")

def inspection_id(): return f"INS-{utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

def user_view(u):
    return {k:v for k,v in clean(u).items() if k not in {"password_hash"}}

@router.post("/auth/login", summary="Login using an administrator-created account")
def login(payload:dict):
    ensure_mongo(); email=str(payload.get("email","")).strip().lower(); password=str(payload.get("password", ""))
    user=db().users.find_one({"email":email})
    if not user or not user.get("is_active") or not verify_password(password,user["password_hash"]): raise HTTPException(401,"Invalid email or password.")
    db().users.update_one({"_id":user["_id"]},{"$set":{"last_login":utcnow()}}); audit(user,"LOGIN")
    return {"access_token":token_for(user),"token_type":"bearer","user":user_view(user)}

@router.get("/auth/me")
def me(user=Depends(current_user)): return user_view(user)

@router.get("/admin/dashboard")
def admin_dashboard(admin=Depends(require_admin)):
    ensure_mongo(); d=db(); total=d.inspections.count_documents({}); defects=d.inspections.count_documents({"final_label":"DEFECT"}); pending=d.inspections.count_documents({"review_required":True,"review_completed":False})
    return {"total_inspections":total,"total_ok":d.inspections.count_documents({"final_label":"OK"}),"total_defect":defects,"defect_rate":round(defects/total*100,2) if total else 0,"pending_human_reviews":pending,"supervisors":d.users.count_documents({"role":"SUPERVISOR","is_active":True}),"active_machines":d.machines.count_documents({"status":"ACTIVE"}),"current_model":d.model_versions.find_one({"status":"PRODUCTION"},{"_id":0})}

@router.get("/admin/team")
def team(admin=Depends(require_admin)):
    ensure_mongo(); return [user_view(x) for x in db().users.find({"role":"SUPERVISOR"}).sort("created_at",-1)]

@router.get("/admin/inspections")
def all_inspections(machine_id:str|None=None, admin=Depends(require_admin)):
    ensure_mongo(); q={"machine_id":machine_id} if machine_id else {}; return [clean(x) for x in db().inspections.find(q).sort("timestamp",-1).limit(500)]

@router.get("/admin/team/supervisors")
def supervisors(admin=Depends(require_admin)):
    ensure_mongo(); return [user_view(x) for x in db().users.find({"role":"SUPERVISOR"}).sort("created_at",-1)]

@router.post("/admin/team/supervisors",status_code=201)
def create_supervisor(payload:dict,admin=Depends(require_admin)):
    ensure_mongo(); required=["full_name","email","employee_id","password","machine_id"]
    if any(not str(payload.get(k,"")).strip() for k in required): raise HTTPException(422,"Full name, email, employee ID, password, and machine assignment are required.")
    machine=db().machines.find_one({"machine_id":payload["machine_id"]})
    if not machine: raise HTTPException(422,"Assigned machine does not exist.")
    doc={"_id":uuid.uuid4().hex,"full_name":payload["full_name"].strip(),"email":payload["email"].strip().lower(),"employee_id":payload["employee_id"].strip(),"password_hash":hash_password(payload["password"]),"role":"SUPERVISOR","machine_id":payload["machine_id"],"is_active":bool(payload.get("is_active",True)),"created_at":utcnow(),"updated_at":utcnow()}
    try: db().users.insert_one(doc)
    except DuplicateKeyError: raise HTTPException(409,"Email or employee ID already exists.")
    audit(admin,"SUPERVISOR_CREATED",doc["_id"],{"machine_id":doc["machine_id"]}); return user_view(doc)

@router.put("/admin/team/supervisors/{user_id}")
def update_supervisor(user_id:str,payload:dict,admin=Depends(require_admin)):
    ensure_mongo(); allowed={k:payload[k] for k in ("full_name","employee_id","is_active","machine_id") if k in payload}
    if "machine_id" in allowed and not db().machines.find_one({"machine_id":allowed["machine_id"]}): raise HTTPException(422,"Assigned machine does not exist.")
    if not allowed: raise HTTPException(422,"No supported fields supplied.")
    allowed["updated_at"]=utcnow(); result=db().users.find_one_and_update({"_id":user_id,"role":"SUPERVISOR"},{"$set":allowed},return_document=True)
    if not result: raise HTTPException(404,"Supervisor not found.")
    audit(admin,"SUPERVISOR_UPDATED",user_id,allowed); return user_view(result)

@router.delete("/admin/team/supervisors/{user_id}")
def deactivate_supervisor(user_id:str,admin=Depends(require_admin)):
    ensure_mongo(); result=db().users.update_one({"_id":user_id,"role":"SUPERVISOR"},{"$set":{"is_active":False,"updated_at":utcnow()}})
    if not result.matched_count: raise HTTPException(404,"Supervisor not found.")
    audit(admin,"SUPERVISOR_DEACTIVATED",user_id); return {"status":"deactivated"}

@router.post("/admin/team/supervisors/{user_id}/reset-password")
def reset_password(user_id:str,payload:dict,admin=Depends(require_admin)):
    password=str(payload.get("password", "")); hashed=hash_password(password); result=db().users.update_one({"_id":user_id,"role":"SUPERVISOR"},{"$set":{"password_hash":hashed,"updated_at":utcnow()}})
    if not result.matched_count: raise HTTPException(404,"Supervisor not found.")
    audit(admin,"SUPERVISOR_PASSWORD_RESET",user_id); return {"status":"password_reset"}

@router.get("/admin/machines")
def machines(admin=Depends(require_admin)): ensure_mongo(); return [clean(x) for x in db().machines.find().sort("machine_code",1)]
@router.post("/admin/machines",status_code=201)
def create_machine(payload:dict,admin=Depends(require_admin)):
    ensure_mongo(); code=str(payload.get("machine_code","")).strip().upper(); name=str(payload.get("name","")).strip()
    if not code or not name: raise HTTPException(422,"Machine code and name are required.")
    doc={"_id":uuid.uuid4().hex,"machine_id":uuid.uuid4().hex,"machine_code":code,"name":name,"description":str(payload.get("description", "")),"status":payload.get("status","ACTIVE"),"created_at":utcnow(),"updated_at":utcnow()}
    try: db().machines.insert_one(doc)
    except DuplicateKeyError: raise HTTPException(409,"Machine code already exists.")
    audit(admin,"MACHINE_CREATED",doc["machine_id"]); return clean(doc)
@router.put("/admin/machines/{machine_id}")
def update_machine(machine_id:str,payload:dict,admin=Depends(require_admin)):
    ensure_mongo(); changes={k:payload[k] for k in ("name","description","status","machine_code") if k in payload}; changes["updated_at"]=utcnow()
    rec=db().machines.find_one_and_update({"machine_id":machine_id},{"$set":changes},return_document=True)
    if not rec: raise HTTPException(404,"Machine not found.")
    audit(admin,"MACHINE_UPDATED",machine_id,changes); return clean(rec)

@router.get("/supervisor/dashboard")
def supervisor_dashboard(user=Depends(require_supervisor)):
    ensure_mongo(); q={"machine_id":user["machine_id"]}; d=db(); total=d.inspections.count_documents(q); defects=d.inspections.count_documents(q|{"final_label":"DEFECT"})
    return {"machine":clean(d.machines.find_one({"machine_id":user["machine_id"]})),"today_inspections":total,"today_defects":defects,"pending_reviews":d.inspections.count_documents(q|{"review_required":True,"review_completed":False}),"recent_inspections":[clean(x) for x in d.inspections.find(q).sort("timestamp",-1).limit(10)]}
@router.get("/supervisor/machine")
def supervisor_machine(user=Depends(require_supervisor)): ensure_mongo(); return clean(db().machines.find_one({"machine_id":user["machine_id"]}))

@router.post("/supervisor/inspect",status_code=201)
async def inspect(image:UploadFile=File(...), product_category:str|None=Form(None), user=Depends(require_supervisor)):
    ensure_mongo()
    if image.content_type not in ALLOWED: raise HTTPException(415,"Only JPEG, PNG, and WEBP images are accepted.")
    body=await image.read(settings.MAX_UPLOAD_BYTES + 1)
    if not body: raise HTTPException(400,"Image is empty.")
    if len(body)>settings.MAX_UPLOAD_BYTES: raise HTTPException(413,"Image exceeds the configured 10 MB limit.")
    iid=inspection_id(); storage=Path(settings.IMAGE_STORAGE_PATH); storage.mkdir(parents=True,exist_ok=True); reference=storage/f"{iid}.jpg"; reference.write_bytes(body)
    try: result=await bento.predict(body,filename=f"{iid}{ALLOWED[image.content_type]}",product_category=product_category)
    except Exception: raise HTTPException(503,"Model service unavailable. Try again.")
    prediction=str(result.get("prediction","")).upper(); confidence=float(result.get("confidence",0));
    if prediction not in {"OK","DEFECT"} or not 0<=confidence<=1: raise HTTPException(503,"Model service returned an invalid prediction.")
    outcome=decide(prediction,confidence); doc={"_id":uuid.uuid4().hex,"inspection_id":iid,"machine_id":user["machine_id"],"supervisor_id":user["_id"],"timestamp":utcnow(),"image_reference":str(reference),"prediction":prediction,"predicted_label":prediction,"confidence":confidence,"model_name":result.get("model_name", "ResNet-50"),"model_version":result.get("model_version","unknown"),"inference_latency_ms":float(result.get("latency_ms",0)),"status":outcome.status,"decision":outcome.decision,"final_label":outcome.final_label,"review_required":outcome.review_required,"review_completed":False,"operator_flagged":False,"operator_action":None,"operator_remark":None}
    db().inspections.insert_one(doc); audit(user,"INSPECTION_PERFORMED",iid,{"prediction":prediction,"confidence":confidence}); return clean(doc)

@router.get("/supervisor/inspections")
def supervisor_inspections(search:str|None=None,status_filter:str|None=Query(None,alias="status"),user=Depends(require_supervisor)):
    ensure_mongo(); q={"machine_id":user["machine_id"]};
    if search: q["inspection_id"]={"$regex":search,"$options":"i"}
    if status_filter: q["status"]=status_filter
    return [clean(x) for x in db().inspections.find(q).sort("timestamp",-1).limit(200)]
@router.get("/supervisor/inspections/{inspection_id}")
def supervisor_inspection(inspection_id:str,user=Depends(require_supervisor)):
    ensure_mongo(); rec=db().inspections.find_one({"inspection_id":inspection_id})
    if not rec: raise HTTPException(404,"Inspection not found.")
    if rec["machine_id"]!=user["machine_id"]: raise HTTPException(403,"Inspection belongs to another machine.")
    return clean(rec)
@router.post("/supervisor/inspections/{inspection_id}/flag")
def flag(inspection_id:str,user=Depends(require_supervisor)):
    ensure_mongo(); rec=db().inspections.find_one_and_update({"inspection_id":inspection_id,"machine_id":user["machine_id"],"review_completed":False},{"$set":{"review_required":True,"operator_flagged":True,"status":"FLAGGED","decision":"PENDING_HUMAN_REVIEW","final_label":None}},return_document=True)
    if not rec: raise HTTPException(404,"Open inspection not found.")
    audit(user,"INSPECTION_FLAGGED",inspection_id); return clean(rec)
@router.post("/supervisor/inspections/{inspection_id}/review")
def review(inspection_id:str,payload:dict,user=Depends(require_supervisor)):
    ensure_mongo(); label=str(payload.get("human_label","")).upper(); remark=str(payload.get("remark","")).strip()
    if label not in {"OK","DEFECT"} or not remark: raise HTTPException(422,"A final label and non-empty remark are required.")
    rec=db().inspections.find_one({"inspection_id":inspection_id,"machine_id":user["machine_id"]})
    if not rec: raise HTTPException(404,"Inspection not found.")
    if rec.get("review_completed"): raise HTTPException(409,"This inspection has already been reviewed.")
    # High-confidence DEFECT approval/denial and every pending/flagged review converge here.
    final_status="HUMAN_CONFIRMED_"+label
    db().inspections.update_one({"_id":rec["_id"]},{"$set":{"final_label":label,"human_label":label,"operator_action":label,"operator_remark":remark,"review_required":True,"review_completed":True,"status":final_status,"decision":"HUMAN_APPROVED" if label=="DEFECT" else "HUMAN_REJECTED"}})
    feedback={"_id":uuid.uuid4().hex,"feedback_id":uuid.uuid4().hex,"inspection_id":inspection_id,"image_reference":rec["image_reference"],"machine_id":rec["machine_id"],"supervisor_id":user["_id"],"predicted_label":rec["predicted_label"],"human_label":label,"confidence":rec["confidence"],"remark":remark,"model_version":rec["model_version"],"created_at":utcnow(),"included_in_training":False,"dataset_version":None}
    db().feedback.insert_one(feedback); audit(user,"HUMAN_REVIEW_SUBMITTED",inspection_id,{"human_label":label}); return clean(db().inspections.find_one({"_id":rec["_id"]}))

@router.get("/feedback/pending")
def pending_feedback(user=Depends(current_user)):
    ensure_mongo(); q={"review_required":True,"review_completed":False};
    if user["role"]=="SUPERVISOR": q["machine_id"]=user["machine_id"]
    return [clean(x) for x in db().inspections.find(q).sort("timestamp",1)]
@router.get("/admin/feedback")
def feedback_pool(admin=Depends(require_admin)): ensure_mongo(); return [clean(x) for x in db().feedback.find().sort("created_at",-1)]
@router.post("/admin/feedback/{feedback_id}/approve")
def approve_feedback(feedback_id:str,payload:dict|None=None,admin=Depends(require_admin)):
    ensure_mongo(); rec=db().feedback.find_one_and_update({"feedback_id":feedback_id},{"$set":{"included_in_training":True,"dataset_version":(payload or {}).get("dataset_version")}},return_document=True)
    if not rec: raise HTTPException(404,"Feedback not found.")
    audit(admin,"FEEDBACK_APPROVED_FOR_TRAINING",feedback_id); return clean(rec)

@router.get("/models")
def models(user=Depends(current_user)): ensure_mongo(); return [clean(x) for x in db().model_versions.find().sort("created_at",-1)]
@router.get("/models/active")
def active_model(user=Depends(current_user)):
    ensure_mongo(); return clean(db().model_versions.find_one({"status":"PRODUCTION"})) or {"status":"NOT_AVAILABLE"}
@router.get("/retraining/status")
def retraining_status(admin=Depends(require_admin)):
    ensure_mongo(); latest=db().retraining_jobs.find_one(sort=[("started_at",-1)]); return clean(latest) or {"status":"IDLE","feedback_count":db().feedback.count_documents({"included_in_training":True})}
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
