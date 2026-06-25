from typing import Callable

class JobQueue:
    def process_job(self, job_id: str, processor: Callable) -> bool:
        delays = [1, 2, 4]
        job_data = None
        if hasattr(self, 'jobs') and isinstance(self.jobs, dict):
            job_data = self.jobs.get(job_id)
        if job_data is None:
            if hasattr(self, '_jobs') and isinstance(self._jobs, dict):
                job_data = self._jobs.get(job_id)
        if job_data is None:
            if hasattr(self, 'job_data') and isinstance(self.job_data, dict):
                job_data = self.job_data.get(job_id)
        if job_data is None:
            for attr in dir(self):
                if not attr.startswith('_'):
                    val = getattr(self, attr)
                    if isinstance(val, dict) and job_id in val:
                        job_data = val[job_id]
                        break
        if job_data is None:
            job_data = {}

        attempt = 0
        delay_idx = 0
        while attempt < 4:
            attempt += 1
            try:
                processor(job_data)
                return True
            except Exception:
                if attempt < 4:
                    delay = delays[delay_idx]
                    delay_idx += 1
                    if 'delays' not in job_data:
                        job_data['delays'] = []
                    job_data['delays'].append(delay)
                else:
                    return False
        return False