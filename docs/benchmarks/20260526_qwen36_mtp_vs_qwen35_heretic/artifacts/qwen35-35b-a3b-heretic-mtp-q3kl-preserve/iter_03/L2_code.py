queue = JobQueue()
        queue.add_job("job1", {"url": "https://example.com"})
        # ...
        success = queue.process_job("job1", fetch_url)