class JobQueue:
    def __init__(self):
        self._queue = []
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def process(self) -> dict | None:
        if self._queue:
            job_id, data = self._queue.pop(0)
            result = {"status": "completed", "data": data}
            self._results[job_id] = result
            return result
        return None

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)