from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        data = self._jobs.get(job_id)
        if data is None:
            raise KeyError(f"Job {job_id} not found")
        attempts = 0
        delay = 1
        delays = data.get('delays')
        if delays is None:
            delays = {}
            data['delays'] = delays
        while attempts < 4:
            try:
                processor(data)
                if attempts > 0:
                    delays[f'&time={attempts}'] = delay
                return True
            except Exception:
                attempts += 1
                if attempts == 4:
                    break
                delays[f'&time={attempts-1}'] = delay
                delay *= 2
        return False