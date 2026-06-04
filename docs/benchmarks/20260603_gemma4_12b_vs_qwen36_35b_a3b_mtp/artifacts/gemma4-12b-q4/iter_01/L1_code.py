import threading
    import time
    from queue import Queue
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.results: Dict[str, dict] = {}
            self.queue = Queue()
            self.worker = threading.Thread(target=self._process_loop, daemon=True)
            self.worker.start()

        def _process_loop(self):
            while True:
                job_id, data = self.queue.get()
                # Simulate processing
                time.sleep(0.1)
                # Simulate a result
                self.results[job_id] = {"status": "completed", "data": data}
                self.queue.task_done()

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.put((job_id, data))
            return job_id

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)