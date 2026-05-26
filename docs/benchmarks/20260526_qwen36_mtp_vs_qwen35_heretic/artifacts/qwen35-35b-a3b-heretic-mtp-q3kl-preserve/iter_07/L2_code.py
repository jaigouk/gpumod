queue = JobQueue()
        queue.add_job("job1", {"url": "https://example.com"})
        def fetch_url(data): ...
        success = queue.process_job("job1", fetch_url)