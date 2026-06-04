from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append((job_id, data))
            return job_id

        def get_result(self, job_id: str) -> Optional[dict]:
            return self.results.get(job_id)

        # To make it actually usable for a test, I'll add a completion method
        # but the prompt says "Write only the Python code" based on requirements.
        # If I add an extra method, is it okay?
        # Let's look at the requirements again. Only 2 methods required.