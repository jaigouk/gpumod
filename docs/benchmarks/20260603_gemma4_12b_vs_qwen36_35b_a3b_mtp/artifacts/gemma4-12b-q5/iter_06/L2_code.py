from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data
        self.jobs[job_id]].setdefault('retry_count', 0)
        self.jobs[job_id]].setdefault('recorded_delays', [])

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            return False

        for attempt in range(4):
            try:
                Processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    data['retry_count'] = data.get('retry_count', 0) + 1
                    data['recorded_delays'] = data.get('recorded_delays', []) + [delay]
                else:
                    return False
        return False