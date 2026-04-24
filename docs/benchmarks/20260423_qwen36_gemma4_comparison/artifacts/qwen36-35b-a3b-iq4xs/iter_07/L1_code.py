class JobQueue:
    def __init__(self):
        self._queue = []
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def process(self):
        while self._queue:
            job_id, data = self._queue.pop(0)
            self._results[job_id] = {"status": "completed", "data": data}

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id, None)