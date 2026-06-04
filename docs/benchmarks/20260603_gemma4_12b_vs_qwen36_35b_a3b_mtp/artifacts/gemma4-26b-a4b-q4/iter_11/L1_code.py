import threading
    import queue
    import time

    class JobQueue:
        def __init__(self):
            self.jobs = queue.Queue()
            self.results = {}
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()

        def _worker(self):
            while True:
                job_id, data = self.jobs.get()
                # Simulate processing
                time.sleep(0.1)
                result = {"status": "completed", "original_data": data}
                self.results[job_id] = result
                self.jobs.task_done()

        def add_job(self, job_id: str, data: dict) -> str:
            self.jobs.put((job_id, data))
            return job_id

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)