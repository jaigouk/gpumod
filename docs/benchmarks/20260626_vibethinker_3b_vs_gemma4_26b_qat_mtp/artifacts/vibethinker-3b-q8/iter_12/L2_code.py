from typing import Callable, Dict, Any
import collections

class JobQueue:
    def __init__(self) -> None:
        self.queue: collections.deque = collections.deque()
        self.data_by_id: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}
        self.delay_log: Dict[str, Dict[int, float]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.data_by_id[job_id] = data
        self.queue.append((job_id, data))
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        data = self.data_by_id.get(job_id)
        if not data:
            return False

        attempts_made = self.retry_counts.get(job_id, 0)
        if attempts_made >= 4:
            return False

        for i in range(attempts_made, 4):
            attempt_no = i + 1
            delay = 2 ** (attempt_no - 2) if attempt_no > 1 else 0
            cur_data = data.copy()
            if attempt_no > 1:
                cur_data['delay'] = delay
            try:
                result = processor(cur_data)
                self.retry_counts[job_id] = attempt_no
                del self.data_by_id[job_id]
                for idx, item in enumerate(self.queue):
                    if item[0] == job_id:
                        self.queue.pop(idx)
                        break
                return True
            except Exception:
                if job_id not in self.delay_log:
                    self.delay_log[job_id] = {}
                self.delay_log[job_id][attempt_no] = delay

        self.retry_counts[job_id] = 4
        self.data_by_id.pop(job_id, None)
        return False