import asyncio
import json
from collections import defaultdict


class EventBus:
    """In-process fallback; Redis Streams can replace this without changing API contracts."""
    def __init__(self):
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def publish(self, topic: str, payload: dict) -> None:
        for queue in list(self._queues[topic]):
            await queue.put(payload)

    async def subscribe(self, topic: str):
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[topic].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._queues[topic].remove(queue)


def sse(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
