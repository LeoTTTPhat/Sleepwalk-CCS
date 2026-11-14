#!/usr/bin/env python3
"""Stand-in for the memory guard that wrapped every heavy step on our shared machine.

The run scripts call  memguard.py --name NAME [guard options] -- COMMAND ...
This stand-in accepts the same arguments, ignores the guard options and runs COMMAND directly, returning its exit code.
Set MEMGUARD to the path of another wrapper to use one.
"""
import subprocess
import sys

argv = sys.argv[1:]
if "--" not in argv:
    sys.exit("usage: memguard.py [options] -- command ...")
cmd = argv[argv.index("--") + 1:]
if not cmd:
    sys.exit("memguard.py: no command after --")
sys.exit(subprocess.call(cmd))
