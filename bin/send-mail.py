#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""
This script tests sending an email from Python to test the localhost SMTP server
is correctly configured.
"""

import getpass
import argparse
from smtplib import SMTP

# Gather our code in a main() function
def main():
  parser = argparse.ArgumentParser(
    description='Program to send a test e-mail to the given addresses.')
  parser.add_argument('address', nargs='+', help='e-mail recipient address')

  group = parser.add_argument_group('Message')
  group.add_argument('--sender', dest='sender',
    default=getpass.getuser() + '@sussex.ac.uk',
    help='e-mail sender [%(default)s]')
  group.add_argument('-s', '--subject', dest='subject',
    default='SMTP e-mail test',
    help='e-mail subject [%(default)s]')
  group.add_argument('-m', '--message', dest='message',
    default=('This is a test e-mail message.'),
    help='e-mail message [%(default)s]')
  group.add_argument('-f', '--filename', dest='file', metavar='FILE',
    default=None,
    help='e-mail message text file (overrides other arguments)')

  args = parser.parse_args()

  from_addr = args.sender
  to_addrs = []

  for arg in args.address:
    to_addrs.append(arg)

  if args.file:
    with open(args.file, 'r') as file:
      message = file.read()
  else:
    message = f"""Subject: {args.subject}

{args.message}
"""

  with SMTP("localhost") as smtp:
    # This method will return normally if the mail is accepted for at
    # least one recipient (i.e. someone should get your mail).
    # Otherwise it will raise an exception.
    smtp.sendmail(from_addr, to_addrs, message)

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()
