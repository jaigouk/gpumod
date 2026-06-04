from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append((job_id, data))
            self.results[job_id] = None
            return job_id

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)

        def complete_job(self, job_id: str, result: dict) -> None:
            self.results[job_id] = result
            # Logic for removing from queue could go here, 
            # but since it's FIFO, we'd need to know which one 
            # was completed to pop from front. 
            # But "processed in FIFO order" usually means we take from front.