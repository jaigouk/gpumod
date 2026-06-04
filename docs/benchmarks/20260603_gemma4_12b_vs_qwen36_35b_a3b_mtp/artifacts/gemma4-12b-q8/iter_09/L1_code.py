import threading
    import queue
    import time

    class JobQueue:
        def __init__(self):
            self.queue = queue.Queue()
            self.results = {}
            self.lock = threading.Lock()

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.put((job_id, data))
            return job_id

        def get_result(self, job_id: str) -> dict | None:
            with self.lock:
                return self.results.get(job_id)

        def _worker(self):
            while True:
                job_id, data = self.queue.get()
                # Simulate processing
                time.sleep(1)
                with self.lock:
                    self.results[job_id] = {"status": "completed", "data": data}
                self.queue.task_done()

        def start(self):
            threading.Thread(target=self._worker, daemon=True).start()