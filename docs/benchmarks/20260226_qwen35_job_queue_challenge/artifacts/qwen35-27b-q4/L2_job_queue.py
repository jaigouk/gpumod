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
        self._queue.put((fn, args, kwargs, job_id, 0))
        return job_id

    def get_result(self, job_id):
        with self._lock:
            if job_id not in self._results:
                return None
            result = self._results[job_id]
            if isinstance(result, Exception):
                raise result
            return result

    def _worker_loop(self):
        while True:
            try:
                fn, args, kwargs, job_id, retry_count = self._queue.get()
                try:
                    result = fn(*args, **kwargs)
                    with self._lock:
                        self._results[job_id] = result
                    self._queue.task_done()
                except Exception as e:
                    retry_count += 1
                    if retry_count <= 3:
                        wait_time = 2 ** (retry_count - 1)
                        time.sleep(wait_time)
                        self._queue.put((fn, args, kwargs, job_id, retry_count))
                    else:
                        with self._lock:
                            self._results[job_id] = e
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
        try:
            result = jq.get_result(job_id)
        except Exception as e:
            print(f"Job failed: {e}")
            break
        time.sleep(0.1)
        
    if result is not None:
        print(f"Result: {result}")
