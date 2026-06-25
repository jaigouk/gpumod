from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: Dict) -> None:
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[Dict], Any]) -> bool:
        data = self._jobs.pop(job_id, None)
        if data is None:
            return False

        attempts = 0
        while attempts < 4:
            attempts += 1
            try:
                processor(data)
                data['retry_count'] = attempts
                return True
            except:
                if attempts < 4:
                    delay = 2 ** (attempts - 1)
                    data[f'delay_{attempts}'] = delay
                data['retry_count'] = attempts
        data['retry_count'] = 4
        return False