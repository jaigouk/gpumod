import collections

class JobQueue:
    def __init__(self):
        self.job_queue = collections.deque()
        self.results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.job_queue.append({'job_id': job_id, 'data': data, 'result': None, 'status': 'pending'})
        self.results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self.results.get(job_id)

    def set_result(self, job_id: str, result: dict):
        self.results[job_id] = result
        for job in self.job_queue:
            if job['job_id'] == job_id:
                job['result'] = result
                job['status'] = 'completed'
                break

    def process_next(self):
        while self.job_queue and self.job_queue[0]['status'] != 'pending':
            self.job_queue.popleft()

        if not self.job_queue:
            return None

        job = self.job_queue.popleft()
        job['status'] = 'completed'
        job['result'] = {'status': 'processed', 'data': job['data']}
        self.results[job['job_id']] = job['result']
        return job['job_id']