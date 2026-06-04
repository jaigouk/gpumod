max_retries = 3
        retries_done = self.jobs[job_id]['retries']
        for attempt in range(max_retries + 1):
            try:
                processor(self.jobs[job_id]['data'])
                self.jobs[job_id]['retries'] = 0 # Reset or keep? Usually reset on success.
                return True
            except Exception:
                if attempt < max_retries:
                    self.jobs[job_id]['retries'] += 1
                    backoff = 2 ** (self.jobs[job_id]['retries'] - 1)
                    # Log backoff or just continue
                else:
                    return False