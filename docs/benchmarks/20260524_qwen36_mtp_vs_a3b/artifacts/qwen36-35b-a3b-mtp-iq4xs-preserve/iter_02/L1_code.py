import collections

class JobQueue:
    def __init__(self):
        self._queue = collections.deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append(job_id)
        # Store data if needed, or just job_id
        if not hasattr(self, '_data'):
            self._data = {}
        self._data[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id, None)

    def process(self) -> dict | None:
        if not self._queue:
            return None
        job_id = self._queue.popleft()
        # Simulate processing
        self._results[job_id] = {"status": "completed", "data": self._data.get(job_id, {})}
        return self._results[job_id]