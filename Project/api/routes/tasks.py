from fastapi import APIRouter, HTTPException

from schemas.api import TaskResponse
from jobs.tasks import get_task, list_tasks

router = APIRouter()


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task_status(task_id: str):
    rec = get_task(task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="task not found")
    return rec


@router.get("/tasks", response_model=list[TaskResponse])
def get_task_list(limit: int = 20, offset: int = 0):
    return list_tasks(limit=limit, offset=offset)
