class Job:
       def __init__(self, job_id, data):
           self.job_id = job_id
           self.data = data
           self.retry_count = 0
           self.backoff_delays = []