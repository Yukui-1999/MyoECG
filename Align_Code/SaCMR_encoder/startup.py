"""Startup progress that remains visible while the caller waits for storage."""
import json
import sys
import threading
import time


class StartupProgress:
    def __init__(self, interval=10., stream=None):
        self.interval = interval
        self.stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stage = 'starting'
        self._details = {}
        self._started = time.monotonic()
        self._thread = None
        self._finished = False

    def _emit(self, event):
        with self._lock:
            record = dict(self._details, event=event, stage=self._stage,
                          elapsed_seconds=round(time.monotonic() - self._started, 1))
            print(json.dumps(record, ensure_ascii=False, default=str),
                  file=self.stream, flush=True)

    def update(self, stage, **details):
        with self._lock:
            changed = stage != self._stage
            self._stage = stage
            self._details = details
        if changed:
            self._emit('startup')

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self._emit('heartbeat')

    def __enter__(self):
        self._thread = threading.Thread(target=self._heartbeat, daemon=True,
                                        name='sa-startup-progress')
        self._thread.start()
        return self

    def stop(self, event='startup_complete'):
        if self._finished:
            return
        self._finished = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._emit(event)

    def __exit__(self, exc_type, exc, traceback):
        self.stop('startup_failed' if exc_type is not None else 'startup_complete')
