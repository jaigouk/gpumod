attempts = 0
                max_retries = 3
                while attempts <= max_retries:
                    try:
                        return processor(job_data)
                    except Exception:
                        attempts += 1
                        if attempts <= max_retries:
                            delay = backoff_list[attempts-1]
                            # track delay
                            time.sleep(delay)
                        else:
                            return False