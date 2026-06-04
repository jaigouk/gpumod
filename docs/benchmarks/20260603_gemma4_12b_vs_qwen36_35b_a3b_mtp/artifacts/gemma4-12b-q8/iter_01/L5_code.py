from dataclasses import dataclass
        from typing import Any, Optional
        from datetime import datetime

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0
            created_at: datetime = datetime.now()