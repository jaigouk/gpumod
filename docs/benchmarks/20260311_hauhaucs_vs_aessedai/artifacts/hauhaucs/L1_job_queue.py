import threading
import queue
import uuid

class JobQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._results = {}
        self._lock = threading.Lock()
        self._running = True
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
        while self._running:
            try:
                fn, args, kwargs, job_id = self._queue.get(timeout=0.5)
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    result = e
                with self._lock:
                    self._results[job_id] = result
                self._queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._running = False
        self._worker.join()

if __name__ == "__main__":
    import time

    def sample_task(x):
        time.sleep(1)
        return x * 2

    q = JobQueue()
    job_id = q.add_job(sample_task, 5)
    print(f"Job added: {job_id}")

    while True:
        res = q.get_result(job_id)
        if res is not None:
            print(f"Result: {res}")
            break
        time.sleep(0.1)

    q.stop()
