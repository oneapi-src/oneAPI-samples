# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import json
import time
import threading
from contextlib import contextmanager

try:
  from queue import Queue
except ImportError:
  from Queue import Queue


def to_microseconds(s):
  return 1000000 * float(s)


class TraceWriter(threading.Thread):

  def __init__(self, terminator, input_queue, output_stream):
    threading.Thread.__init__(self)
    self.daemon = True
    self.terminator = terminator
    self.input = input_queue
    self.output = output_stream

  def _open_collection(self):
    """Write the opening of a JSON array to the output."""
    self.output.write(b'[')

  def _close_collection(self):
    """Write the closing of a JSON array to the output."""
    self.output.write(b'{}]')  # empty {} so the final entry doesn't end with a comma

  def run(self):
    self._open_collection()
    while not self.terminator.is_set() or not self.input.empty():
      item = self.input.get()
      self.output.write((json.dumps(item) + ',\n').encode('ascii'))
    self._close_collection()


class TraceProfiler(object):
  """A python trace profiler that outputs Chrome Trace-Viewer format (about://tracing).

     Usage:

        from pytracing import TraceProfiler
        tp = TraceProfiler(output=open('/tmp/trace.out', 'wb'))
        with tp.traced():
          ...

  """
  TYPES = {'call': 'B', 'return': 'E', 'exec': 'X'}

  def __init__(self, output, clock=None):
    self.output = output
    self.clock = clock or time.time
    self.pid = os.getpid()
    self.queue = Queue()
    self.terminator = threading.Event()
    self.writer = TraceWriter(self.terminator, self.queue, self.output)

  @property
  def thread_id(self):
    return threading.current_thread().name

  @contextmanager
  def traced(self):
    """Context manager for install/shutdown in a with block."""
    self.install()
    try:
      yield
    finally:
      self.shutdown()

  def install(self):
    """Install the trace function and open the JSON output stream."""
    self.writer.start()               # Start the writer thread.

  def shutdown(self):
    self.terminator.set()              # Stop the writer thread.
    self.writer.join()                 # Join the writer thread.

  def fire_event(self, event_type, event_name, event_cat, kernel_name, timestamp, duration, pass_type):
    """Write a trace event to the output stream."""
    # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
    event = dict(
      name=event_name,                 # Event Name.
      cat=event_cat,               # Event Category.
      tid=self.thread_id,             # Thread ID.
      ph=self.TYPES[event_type],      # Event Type.
      pid=self.pid,                   # Process ID.
      ts=timestamp,                   # Timestamp.
      dur=duration,
      args=dict(
        jit=kernel_name,
        pass_type=pass_type,
        )
      )
    self.queue.put(event)

