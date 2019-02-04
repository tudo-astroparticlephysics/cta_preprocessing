import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('input_list')
parser.add_argument('output_directory')
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--sct-data', dest='sct', action='store_true')
feature_parser.add_argument('--no-sct-data', dest='sct', action='store_false')
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--dry-run', dest='dry_run', action='store_true')
feature_parser.add_argument('--no-dry_run', dest='dry_run', action='store_false')
parser.set_defaults(sct=False)
parser.set_defaults(dry_run=False)

args = parser.parse_args()

input_list = args.input_list 

with open(input_list) as f:
    paths = f.readlines()

for i, path in enumerate(paths):
    if args.sct == False and 'SCT' in path:
        continue
    print 'Downloading file {} of {}'.format(i, len(paths))
    basename = os.path.basename(path).strip()
    filename = os.path.join(args.output_directory, basename).strip()
    if os.path.exists(filename):
        print 'File exists. Not downloading {} '.format(basename)
    else:
        if args.dry_run:
            print 'Pretending to download file {} to {}'.format(path.strip(), filename)
            continue
        cmd = ['dirac-dms-get-file', 'LFN:{}'.format(path.strip())]
        subprocess.call(cmd)
        subprocess.call(['mv', basename, filename])
