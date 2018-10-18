import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('input_list')
parser.add_argument('output_directory')
args = parser.parse_args()

input_list = args.input_list 

with open(input_list) as f:
    paths = f.readlines()

for i, path in enumerate(paths):
    print 'Downloading file {} of {}'.format(i, len(paths))
    basename = os.path.basename(path).strip()
    filename = os.path.join(args.output_directory, basename).strip()
    if os.path.exists(filename):
        print 'File exists. Not downloading {} '.format(basename)
    else:
        cmd = ['dirac-dms-get-file', 'LFN:{}'.format(path.strip())]
        subprocess.call(cmd)
        subprocess.call(['mv', basename, filename])
