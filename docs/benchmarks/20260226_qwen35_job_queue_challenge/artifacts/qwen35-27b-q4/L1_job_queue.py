import threading
import queue
import uuid
import time

class JobQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._results = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def add_job(self, fn, *args, **kwargs):
        job_id = str(uuid.uuid4())
        self._queue.put((fn, args, kwargs, job_id))
        return job_id

    def get_result(self, job_id):
        with self._lock:
            return self._results.get(job_id)

    def _worker_loop(self):
        while True:
            try:
                fn, args, kwargs, job_id = self._queue.get()
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    result = e
                with self._lock:
                    self._results[job_id] = result
                self._queue.task_done()
            except Exception:
                break

if __name__ == "__main__":
    jq = JobQueue()
    
    def sample_task(x, y):
        time.sleep(1)
        return x + y

    job_id = jq.add_job(sample_task, 5, 3)
    print(f"Job ID: {job_id}")
    
    result = None
    while result is None:
        result = jq.get_result(job_id)
        time.sleep(0.1)
        
    print(f"Result: {result}")
