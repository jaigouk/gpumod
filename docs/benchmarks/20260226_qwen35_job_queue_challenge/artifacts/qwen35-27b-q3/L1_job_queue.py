import threading
import queue
import uuid
import time

class JobQueue:
    def __init__(self):
        self._job_queue = queue.Queue()
        self._results = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def add_job(self, fn, *args, **kwargs):
        job_id = uuid.uuid4().hex
        self._job_queue.put((fn, args, kwargs, job_id))
        return job_id

    def get_result(self, job_id):
        with self._lock:
            if job_id in self._results:
                return self._results[job_id]
            return None

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                task = self._job_queue.get(timeout=0.1)
                if task is None:
                    break
                fn, args, kwargs, job_id = task
                try:
                    result = fn(*args, **kwargs)
                    with self._lock:
                        self._results[job_id] = result
                except Exception as e:
                    with self._lock:
                        self._results[job_id] = e
                finally:
                    self._job_queue.task_done()
            except queue.Empty:
                continue

    def shutdown(self):
        self._stop_event.set()
        self._worker_thread.join()

if __name__ == "__main__":
    jq = JobQueue()
    
    def sample_job(x, y):
        time.sleep(1)
        return x + y

    id1 = jq.add_job(sample_job, 1, 2)
    id2 = jq.add_job(sample_job, 3, 4)
    
    print(f"Job 1 ID: {id1}")
    print(f"Job 2 ID: {id2}")
    
    time.sleep(2)
    
    print(f"Result 1: {jq.get_result(id1)}")
    print(f"Result 2: {jq.get_result(id2)}")
    
    jq.shutdown()
