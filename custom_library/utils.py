
import time

def get_current_millis():
  return int(round(time.time() * 1000))

def get_elapsed_seconds(current_time, elapsed_millis):
  return (current_time - elapsed_millis) / 1000.0
