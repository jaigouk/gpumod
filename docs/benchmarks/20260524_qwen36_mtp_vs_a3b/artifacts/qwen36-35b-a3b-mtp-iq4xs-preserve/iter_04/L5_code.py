import time
   import random
   from typing import Callable, Any, Optional

   def process_with_retry(func: Callable, max_retries: int = 3, base_delay: float = 1.0, jitter: bool = True) -> Callable[..., Any]:
       # Actually, this is better as a decorator. But the prompt says def process_with_retry(): ...
       # I'll make it a decorator that can also be called directly if needed, or just a decorator.
       # Let's stick to a decorator pattern but name it process_with_retry.
       pass