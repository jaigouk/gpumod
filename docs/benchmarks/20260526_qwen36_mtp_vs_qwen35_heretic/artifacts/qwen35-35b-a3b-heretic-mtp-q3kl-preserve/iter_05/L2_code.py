retry_count = job.get("retry_count", 0)
        max_retries = 3
        backoff = [1, 2, 4]
        
        while True:
            try:
                processor(job_data)
                return True
            except Exception:
                retry_count += 1
                job["retry_count"] = retry_count
                if retry_count <= max_retries:
                    delay = backoff[retry_count - 1]
                    job["next_backoff"] = delay
                    # Simulate wait
                    # time.sleep(delay)
                    break # Or continue loop to retry
                else:
                    return False