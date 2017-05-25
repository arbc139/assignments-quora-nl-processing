
from custom_library.utils import get_current_millis, get_elapsed_seconds

class TimeLogger():
  def __init__(self):
    self.elapsed_millis = 0
  
  def start(self):
    self.elapsed_millis = get_current_millis()
  
  def log_with_elapse(self, log):
    print(log, get_elapsed_seconds(get_current_millis(), self.elapsed_millis))
    
