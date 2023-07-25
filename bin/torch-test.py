#!/usr/bin/env python3

"""
This script tests performing a computation on the GPU using pytorch.
"""

import argparse

CODE_TORCH_GPU = 0
CODE_TORCH_CPU = 1
CODE_NO_TORCH = -1

# Gather our code in a main() function
def main():
  parser = argparse.ArgumentParser(
    description='Program to perform a computation on the GPU using pytorch.',
    epilog='Note: The exit status is non-zero if GPU computation failed.')

  parser.add_argument('-d', '--debug', dest='debug', action='store_true',
    help='Print torch results')
  parser.add_argument('-m', '--memory', dest='memory', action='store_true',
    help='Print memory usage at each step')
  parser.add_argument('-c', '--convolution', dest='convolution',
    action='store_true',
    help='Perform a convolution')

  args = parser.parse_args()

  # Set-up functions for output
  if args.debug:
    # Print out information on torch usage
    def info(msg):
      print(f'TORCH  - {msg}')
  else:
    def info(msg):
      pass
  if args.memory:
    # Print out current resident and virtual memory
    import psutil
    def mem(msg):
      process = psutil.Process()
      i = process.memory_info()
      print(f'MEMORY - {msg}: rss={i.rss/1024**3:.3}G, vms={i.vms/1024**3:.3}G')
  else:
    def mem(msg):
      pass

  mem('start-up')

  code = CODE_NO_TORCH
  try:
    import torch
    mem('import torch')

    gpu = torch.cuda.is_available()
    mem('torch.cuda.is_available()')

    info(f'torch.cuda.is_available() = {gpu}')

    t = torch.rand(5, 3)
    mem('torch.rand(5, 3)')

    if gpu:
      t = t.cuda()
      mem('t = t.cuda()')

    info(t)

    if args.convolution:
      import torch.nn as nn
      mem('import torch.nn')

      m = nn.Conv2d(16, 33, 3, stride=2)
      if gpu:
        m = m.cuda()
      mem('nn.Conv2d')

      t = torch.randn(20, 16, 50, 100)
      if gpu:
        t = t.cuda()
      mem('torch.randn(20, 16, 50, 100)')

      output = m(t)
      mem('convolution done')

      info(output)

    # Reach here if all torch computations worked.
    # Set the exit code to distinguish GPU or CPU
    code = CODE_TORCH_GPU if gpu else CODE_TORCH_CPU

  except Exception as e:
    info(e)
    pass

  # Report if pytorch computation succeeded
  if code == CODE_TORCH_GPU:
    info('OK: torch GPU computation available')
  elif code == CODE_TORCH_CPU:
    info('PASS: torch CPU computation available')
  else:
    info('FAIL: torch GPU computation not available')
  mem(f'exit({code})')
  exit(code)

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()
