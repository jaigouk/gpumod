with self.lock:
            data = self.jobs.pop(job_id, None)
        if data is None:
            return False

        result = processor(data)

        with self.lock:
            self.results[job_id] = result
        return True