"""The Studio's stream: server-sent events to every open tab.

One ``GET /studio/api/stream`` per tab carries every change the page shows
without asking: ``status`` (who holds the GPU, the loaded models, memory),
``workspace`` (a set or a history changed) and, later, jobs and the models
catalog. Each message has an id; the ``Hub`` keeps the latest ones, so a
tab that reconnects with ``Last-Event-ID`` gets what it missed, in order.
A tab that connects fresh, or whose id is no longer kept or belongs to an
earlier process, gets a ``snapshot`` instead: the current state, carrying
the id of the latest message, from which it continues.

Ids are ``<process>-<number>``: the process part changes with every start,
so an id from before a restart is never taken for a recent one. Messages
are published on the event loop; other threads (the worker, the
workspace's) hand them over with ``soon``.
"""

import asyncio
import json
import secrets
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .security import NO_STORE

# Messages kept for replay, and the most a slow tab may fall behind before it
# is cut off (its browser reconnects and replays from the buffer).
KEEP = 10_000
QUEUE = 10_000
PING_SECONDS = 15.0
RETRY_MS = 1000

Message = tuple[str, dict[str, Any]]


class Subscriber:
    """One open stream: the messages waiting to be sent."""

    def __init__(self):
        self.queue: asyncio.Queue[Message | None] = asyncio.Queue(maxsize=QUEUE)
        self.dropped = False

    def put(self, item: Message | None) -> bool:
        """Queue ``item``; False (and the stream ends) when the tab fell too far behind."""
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped = True
            return False
        return True


class Hub:
    """Numbers messages, keeps the latest ``keep`` and hands them to every stream.

    ``snapshot`` returns the current state for a fresh connection.
    Use it from the event loop; ``attach`` names the loop that ``soon``
    hands calls to.
    """

    def __init__(
        self,
        snapshot: Callable[[], dict[str, Any]] = dict,
        keep: int = KEEP,
        ping_seconds: float = PING_SECONDS,
    ):
        self.snapshot = snapshot
        self.ping_seconds = ping_seconds
        self.process = secrets.token_hex(4)
        self.count = 0
        self.recent: deque[tuple[int, Message]] = deque(maxlen=keep)
        self.subscribers: set[Subscriber] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.closed = False
        self._last: dict[str, Any] = {}

    @property
    def last_id(self) -> str:
        return self.id_of(self.count)

    def id_of(self, number: int) -> str:
        return f"{self.process}-{number}"

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def publish(self, kind: str, data: dict[str, Any]) -> str:
        """Send ``{"type": kind, **data}`` to every stream; returns its id."""
        self.count += 1
        item = (self.id_of(self.count), {"type": kind, **data})
        self.recent.append((self.count, item))
        for subscriber in list(self.subscribers):
            if not subscriber.put(item):
                self.subscribers.discard(subscriber)
        return item[0]

    def publish_changed(self, kind: str, data: dict[str, Any]) -> str | None:
        """``publish``, unless the last message of this kind said the same."""
        if self._last.get(kind) == data:
            return None
        self._last[kind] = data
        return self.publish(kind, data)

    def soon(self, function: Callable[..., Any], *args: Any) -> None:
        """Call ``function(*args)`` on the loop, from any thread (dropped when
        there is no loop any more)."""
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        with suppress(RuntimeError):  # the loop closed in between
            loop.call_soon_threadsafe(function, *args)

    def since(self, last_event_id: str | None) -> list[Message] | None:
        """The messages after ``last_event_id``, or None when they are not all
        kept (or the id is unknown): then the stream starts with a snapshot."""
        process, _, number = (last_event_id or "").partition("-")
        if process != self.process or not number.isdigit():
            return None
        number = int(number)
        if number > self.count:
            return None
        oldest = self.recent[0][0] if self.recent else self.count + 1
        if number < oldest - 1:
            return None
        return [item for n, item in self.recent if n > number]

    def subscribe(self, last_event_id: str | None = None) -> tuple[Subscriber, list[Message]]:
        """A new stream and what it sends first: the missed messages, else a snapshot.

        Both are taken in one step on the loop, so nothing published after
        is missed or sent twice.
        """
        subscriber = Subscriber()
        first = self.since(last_event_id)
        if first is None:
            first = [(self.last_id, {"type": "snapshot", **self.snapshot()})]
        self.subscribers.add(subscriber)
        return subscriber, first

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self.subscribers.discard(subscriber)

    def close(self) -> None:
        """End every stream (the Studio is stopping)."""
        self.closed = True
        for subscriber in list(self.subscribers):
            subscriber.put(None)
        self.subscribers.clear()

    async def events(self, subscriber: Subscriber, first: list[Message]) -> AsyncIterator[str]:
        """The stream's text: ``retry``, the first messages, then each one as it
        comes, with a ping comment after ``ping_seconds`` of quiet."""
        try:
            yield f"retry: {RETRY_MS}\n\n"
            for item in first:
                yield encode(*item)
            while not self.closed:
                if subscriber.dropped and subscriber.queue.empty():
                    return
                try:
                    item = await asyncio.wait_for(subscriber.queue.get(), self.ping_seconds)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    return
                yield encode(*item)
        finally:
            self.unsubscribe(subscriber)


def encode(message_id: str, message: dict[str, Any]) -> str:
    data = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    return f"id: {message_id}\ndata: {data}\n\n"


def stream_router(hub: Hub) -> APIRouter:
    router = APIRouter(prefix="/studio/api")

    @router.get("/stream")
    async def stream(request: Request):
        subscriber, first = hub.subscribe(request.headers.get("last-event-id"))
        return StreamingResponse(
            hub.events(subscriber, first),
            media_type="text/event-stream",
            headers={**NO_STORE, "X-Accel-Buffering": "no"},
        )

    return router
