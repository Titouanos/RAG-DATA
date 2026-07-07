"""Suivi des jobs d'ingestion."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from rag_builder.api.deps import current_user, get_db
from rag_builder.api.schemas import JobOut
from rag_builder.db.models import Job, User

router = APIRouter(tags=["jobs"])


def _job_out(j: Job) -> JobOut:
    return JobOut(
        id=j.id,
        collection=j.collection,
        type=j.type,
        status=j.status,
        source_name=j.source_name,
        doc_id=j.doc_id,
        stage=j.stage,
        progress_current=j.progress_current,
        progress_total=j.progress_total,
        message=j.message,
        created_at=j.created_at.isoformat(),
        updated_at=j.updated_at.isoformat(),
    )


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, _user: User = Depends(current_user), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Job introuvable")
    return _job_out(job)


@router.get("/collections/{name}/jobs", response_model=list[JobOut])
def list_jobs(name: str, _user: User = Depends(current_user), db: Session = Depends(get_db)):
    jobs = db.exec(
        select(Job).where(Job.collection == name).order_by(Job.id.desc()).limit(50)
    ).all()
    return [_job_out(j) for j in jobs]
