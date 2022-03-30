import argparse
import sys
import os
from os.path import join as pjoin
import subprocess
import shlex
import platform

hostname = platform.node()

def string_escape(s, encoding='utf-8'):
    """Python 3.x encoding for string escape characters"""
    s = s.replace('\\','/')
    s = s.encode('latin1')
    s = s.decode('unicode-escape')
    # s = s.encode('latin1')
    # print(s)
    # s = s.decode(encoding)
    # print(s)
    return s

class CLIParser(object):
    def __init__(self):

        parser = argparse.ArgumentParser(
            description='Launches the bokeh dashboard for session data visualization')
        parser.add_argument('-a','--animals', action='store',type=str,nargs='+')
        parser.add_argument('-b','--browser', action='store_true',default=False)
    
        args = parser.parse_args(sys.argv[1:])

        cmd = 'bokeh serve'
        if args.browser:
            cmd += ' --show'
        cmdpath = os.path.normpath(pjoin(os.path.abspath(os.path.dirname(__file__)),'bokeh_app.py'))
        cmd += ' ' + string_escape(cmdpath)
        #cmd += ' --host="*"'.encode('string-escape')
        cmd += ' --allow-websocket-origin="*"'#{0}:{1}'.format(hostname,5006)
        cmd += ' --args --animals={0}'.format(','.join([str(a) for a in args.animals]))
        # TODO: PARSE THE ARGUMENTS TO THE BOKEH APP
        print('    ' + cmd,flush=True)
        # c = subprocess.call(['cd', 'C:\\Users\\mouselab\\code\\stimpy\\python_code'], shell=False)
        p = subprocess.call(shlex.split(cmd), shell=False)

def main():
    CLIParser()

if __name__ == '__main__':
    main()

