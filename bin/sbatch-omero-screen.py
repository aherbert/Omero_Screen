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
  send_mail = 'send-mail.py'
  torch_test = 'torch-test.py'

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
  results_dir = os.path.join(args.results_dir, args.username)
  if not os.path.isdir(results_dir):
    raise Exception(f'Missing directory: {results_dir}')

  # Job name
  pid = os.getpid()
  name_width = 14 - len(str(pid))
  name = 'omero_screen'[0:name_width] + '.' + str(pid)

  # Results file to record directory names for screen results
  results_file = f'screen-dir.{name}'
  results_file_path = f'{results_dir}/{results_file}'
  if os.path.isfile(results_file_path):
    raise Exception(f'Results file exists: {results_file_path}')
  open(results_file_path, 'w').close()

  # Create the job file
  script = f'{name}.sh'
  with open(script, 'w') as f:
    # job options
    # The -l option to bash is to make bash act as if a login shell (enables conda init)
    print(inspect.cleandoc(f'''\
      #!/bin/bash -l
      #SBATCH -J {name}
      #SBATCH -o {name}."%j".out
      #SBATCH -p {args.job_class}
      #SBATCH --mail-user {args.username}@sussex.ac.uk
      #SBATCH --mail-type=END,FAIL
      #SBATCH --mem={args.memory}G
      #SBATCH --time={args.hours}:00:00
      '''), file=f)
    if args.threads > 1:
      print(inspect.cleandoc(f'''\
        #SBATCH -n {args.threads}
        '''), file=f)
    if args.gpu:
      # Note: constraint option is not valid although it is specified in the artemis docs
      print(inspect.cleandoc(f'''\
        ##SBATCH --constraint="gpu"
        #SBATCH --gres=gpu
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
      # XXX: Requires '/mnt/lustre/projects/gdsc/conda/miniforge3/bin/conda init bash' to be run in the shell
      #runcmd module add {conda_module}
      runcmd conda activate {conda_env}
      '''.format(run=run, comment=comment, conda_module=conda_module,
                 conda_env=args.env)), file=f)
    # Test for gpu
    if args.gpu:
      print(inspect.cleandoc('''
        set +e
        runcmd python {torch_test}
        code=$?
        if [ $code -ne 0 ]; then
          msg Torch test exit code: $code
          python {send_mail} -m "{msg}" -s "{subject}" {username}@sussex.ac.uk
          exit $code
        fi
        set -e
        '''.format(torch_test=torch_test, send_mail=send_mail,
                   msg=f'Torch GPU unavailable for {script}',
                   subject=f'{script} failed',
                   username=args.username)), file=f);
    for id in set(args.ID):
      debug_flag = '--debug' if args.debug else ''
      print(f'runcmd python {omero_screen_prog} -r {results_dir} '\
        f'-f {results_file} {debug_flag} {id}', file=f)
    # e-mail the user how to collect the results
    # Note: using mailx on the HPC is flakey (either delayed or fails).
    # Here we use a custom python script which sends immediately.
    # Q. How to retrieve data from Artemis? rsync is not allowed on
    # the login nodes.
    subject = f'Job results: {name}'
    msg = f'''
          rsync -ar --files-from=:{results_file_path} {args.username}@ood.artemis.hrc.sussex.ac.uk:{results_dir} .

          Or login (via VPN) to:
            https://ood.artemis.hrc.sussex.ac.uk
          Browse to {results_file_path} via the Files dropdown menu.
          '''
    #print(f'echo \'{msg}\' | mailx -s \'{subject}\' '\
    #  f'{args.username}@sussex.ac.uk', file=f)
    print(f'msg Sending result e-mail using {send_mail}', file=f)
    print(f'python {send_mail} -m \'{msg}\' -s \'{subject}\' '\
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
        https://artemis-docs.hpc.sussex.ac.uk/artemis
      (Requires VPN to access.)'''))
  parser.add_argument('ID', type=int, nargs='+',
    help='Screen ID')
  group = parser.add_argument_group('Job submission')
  group.add_argument('--class', dest='job_class',
    default='gpu',
    help='Job class (default: %(default)s)')
  group.add_argument('--results-dir', dest='results_dir',
    default='/mnt/lustre/users/gdsc/',
    help='Results directory (default: %(default)s)')
  group.add_argument('-u', '--username', dest='username',
    default=getpass.getuser(),
    help='Username (default: %(default)s)')
  group.add_argument('-t', '--threads', type=int, dest='threads',
    default=1,
    help='Threads (default: %(default)s). Use when not executing on the GPU')
  group.add_argument('--hours', type=int,
    default=12,
    help='Expected maximum job hours (default: %(default)s)')
  group.add_argument('-m', '--memory', type=int, dest='memory',
    default=20,
    help='Memory in Gb (default: %(default)s)')
  group.add_argument('--no-gpu', dest='gpu', action='store_false',
    help='Disable using the GPU')
  group.add_argument('--no-exec', dest='exec', action='store_false',
    help='Do not execute script statements. '
      'Use this to submit a job without running Omero Screen')
  group.add_argument('--no-submit', dest='submit', action='store_false',
    help='Do not submit the job script')
  group = parser.add_argument_group('Omero Screen environment')
  group.add_argument('-e', '--env', dest='env', default='omero-screen',
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
    print(subprocess.run(['sbatch', script], stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT, universal_newlines=True).stdout)
