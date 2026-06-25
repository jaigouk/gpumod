import threading
import queue

class JobQueue:
    def __init__(self):
        self._job_queue = queue.Queue()
        self._results = {}
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    def add_job(self, job_id: str, data: dict) -> str:
        self._job_queue.put((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

    def _process_loop(self):
        while True:
            job_id, data = self._job_queue.get()
            self._results[job_id] = {"status": "completed", "data": data}
            self._job_queue.task_done()