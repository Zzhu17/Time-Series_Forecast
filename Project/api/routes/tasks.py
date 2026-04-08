from fastapi import APIRouter, HTTPException

from schemas.api import TaskResponse
from jobs.tasks import get_task, list_tasks, recent_degrade_stats

router = APIRouter()


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task_status(task_id: str):
    rec = get_task(task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="task not found")
    return rec


@router.get("/tasks")
def get_task_list(limit: int = 20, offset: int = 0, stats_window: int = 100):
    items = list_tasks(limit=limit, offset=offset)
    return {
        "items": items,
        "degrade_stats": recent_degrade_stats(window_size=stats_window),
    }
