#!/usr/bin/env python3

import argparse
import inspect
import getpass
import os
import json
import subprocess

def create_job_script(args):
  # Validate installation
  omero_screen = 'Omero_Screen'
  omero_screen_prog = 'omero_screen_run_term'
  conda_module = 'Anaconda3/2022.05'
  send_mail = './send-mail.py'

  if not os.path.isfile(omero_screen_prog):
    raise Exception(f'Missing program: {omero_screen_prog}')
  if not os.path.isfile(send_mail):
    raise Exception(f'Missing program: {send_mail}')
  parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
  if omero_screen != os.path.basename(parent_dir):
    raise Exception(f'Not within an \'{omero_screen}\' installation')
  
  # Check for secrets file
  secrets = '../data/secrets/config.json'
  with open(secrets) as f:
    data = json.load(f)
  for k in ['username', 'password']:
    if not data.get(k):
      raise Exception(f'Secrets file \'{secrets}\' missing key \'{k}\'')

  # Results directory for screen results
  results_dir = '/mnt/lustre/users/gdsc/' + args.username
  if not os.path.isdir(results_dir):
    raise Exception(f'Missing directory: {results_dir}')

  # Job name
  pid = os.getpid()
  name_width = 14 - len(str(pid))
  name = 'omero_screen'[0:name_width] + '.' + str(pid)

  # Results file to record directory names for screen results
  results_file = f'screen-dir.{name}'
  if os.path.isfile(f'{results_dir}/{results_file}'):
    raise Exception(f'Results file exists: {results_dir}/{results_file}')
  open(results_file, 'w').close()

  # Create the job file
  script = f'{name}.sh'
  with open(script, 'w') as f:
    # job options
    print(inspect.cleandoc(f'''\
      #$ -N {name}
      #$ -jc {args.job_class}
      #$ -j Y
      #$ -cwd
      #$ -S /bin/bash
      #$ -M {args.username}@sussex.ac.uk
      #$ -m ea
      '''), file=f)
    if args.threads > 1:
      print(inspect.cleandoc(f'''\
        #$ -q smp.q
        #$ -ps openmp {args.threads}
        #$ -l h_vmem=16G
        '''), file=f)
    if args.gpu:
      print(inspect.cleandoc(f'''\
        #$ -l gpu_card=1
        #$ -l h_vmem=16G
        '''), file=f)
    # job script
    run = 'exec' if args.exec else 'cmd'
    comment = '' if args.exec else '#'
    dollar = '$'
    print(inspect.cleandoc('''
      function msg {{
        echo $(date "+[%F %T]") $@
      }}
      function runcmd {{
        msg {run}: $@
        {comment}$@
      }}
      set -e
      export PYTHONPATH=$(cd ../ && pwd)
      msg PYTHONPATH=$PYTHONPATH
      runcmd module add {conda_module}
      runcmd conda activate {conda_env}
      '''.format(run=run, comment=comment, conda_module=conda_module,
                 conda_env=args.env)), file=f)
    for id in set(args.ID):
      debug_flag = '--debug' if args.debug else ''
      print(f'runcmd python {omero_screen_prog} -r {results_dir} '\
        f'-f {results_file} {debug_flag} {id}', file=f)
    # e-mail the user how to collect the results
    # Note: using mailx on the HPC is flakey (either delayed or fails).
    # Here we use a custom python script which sends immediately.
    subject = f'Job results: {name}'
    msg = f'rsync -a --files-from=:{results_dir}/{results_file} {args.username}'\
          f'@apollo2.hpc.susx.ac.uk:{results_dir} .'
    #print(f'echo \'{msg}\' | mailx -s \'{subject}\' '\
    #  f'{args.username}@sussex.ac.uk', file=f)
    print(f'msg Sending result e-mail using {send_mail}', file=f)
    print(f'{send_mail} -m \'{msg}\' -s \'{subject}\' '\
      f'{args.username}@sussex.ac.uk', file=f)
    print('msg Done', file=f)
    print(f'rm {script}', file=f)
    
    return script
  

def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Program to run Omero Screen on the HPC.',
    epilog=inspect.cleandoc('''Note:
      
      This program makes assumptions on the installation of Omero Screen and
      the run environment.
      
      For HPC information see:
        https://docs.hpc.sussex.ac.uk/apollo2/resources.html'''))
  parser.add_argument('ID', type=int, nargs='+',
    help='Screen ID')
  group = parser.add_argument_group('Job submission')
  group.add_argument('--class', dest='job_class',
    default='test.long',
    help='Job class (default: %(default)s)')
  group.add_argument('-u', '--username', dest='username',
    default=getpass.getuser(),
    help='Username (default: %(default)s)')
  group.add_argument('-t', '--threads', type=int, dest='threads',
    default=1,
    help='Threads (default: %(default)s). Use when not executing on the GPU')
  group.add_argument('--no-gpu', dest='gpu', action='store_false',
    help='Disable using the GPU')
  group.add_argument('--no-exec', dest='exec', action='store_false',
    help='Do not execute script statements. '
      'Use this to submit a job without running Omero Screen')
  group.add_argument('--no-submit', dest='submit', action='store_false',
    help='Do not submit the job script')
  group = parser.add_argument_group('Omero Screen environment')
  group.add_argument('-e', '--env', dest='env', default='cellpose-pip5',
    help='Conda environment (default: %(default)s)')
  group = parser.add_argument_group('Omero Screen overrides')
  group.add_argument('-d', '--debug', dest='debug', action='store_true',
    help='Debug mode')

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  script = create_job_script(args)

  # job submission
  if args.submit:
    print(subprocess.run(['qsub', script], stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT, universal_newlines=True).stdout)
