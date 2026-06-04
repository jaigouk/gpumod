import threading
    import queue
    import time

    class JobQueue:
        def __init__(self):
            self.queue = queue.Queue()
            self.results = {}
            self.lock = threading.Lock()

        def add_job(self, job_id: str, data: dict) -> str:
            with self.lock:
                self.results[job_id] = None
            self.queue.put((job_id, data))
            
            # Start a worker thread for this job to simulate "processing"
            # Or a single worker thread for the whole queue.
            # A single worker is better for FIFO.
            threading.Thread(target=self._worker, args=(job_id, data)).start()
            return job_id

        def _worker(self, job_id, data):
            # Simulate processing
            time.sleep(0.1) 
            with self.lock:
                self.results[job_id] = {"status": "completed", "data": data}

        def get_result(self, job_id: str) -> dict | None:
            with self.lock:
                return self.results.get(job_id)