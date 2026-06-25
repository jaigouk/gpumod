from typing import Callable

class JobQueue:
    def process_job(self, job_id: str, processor: Callable) -> bool:
        job_entry = self.jobs.get(job_id) if hasattr(self, 'jobs') else getattr(self, '_jobs', {}).get(job_id)
        if job_entry is None:
            raise KeyError(f"Job {job_id} not found")
            
        if isinstance(job_entry, dict):
            data = job_entry.get('data')
            if 'delays' not in job_entry:
                job_entry['delays'] = []
            delays_list = job_entry['delays']
        else:
            data = getattr(job_entry, 'data', None)
            if not hasattr(job_entry, 'delays'):
                setattr(job_entry, 'delays', [])
            delays_list = getattr(job_entry, 'delays')
            
        backoff_delays = [1, 2, 4]
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt == 3:
                    return False
                delay = backoff_delays[attempt]
                delays_list.append(delay)
                
        return False