import fact.io
import click
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


@click.command()
@click.argument(
    'input_folder', type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
    ))
@click.argument(
    'output_file', type=click.Path(
        dir_okay=False,
        file_okay=True,
    ))
@click.option('--verify/--no-verify', default=False, help='Wether to verify the file after each append operation')
@click.option('-j', '--n_jobs', default=1, help='number of jobs to use for reading data')
@click.option('-c', '--chunk_size', default=50, help='files per chunk')
def main(input_folder, output_file, verify, n_jobs, chunk_size):
    '''
    process multiple simtel files gievn as INPUT_FILES into one hdf5 file saved in OUTPUT_FILE.
    The hdf5 file will contain three groups. 'runs', 'array_events', 'telescope_events'.

    These files can be put into the classifier tools for learning.
    See https://github.com/fact-project/classifier-tools

    '''
    input_files = glob.glob(f'{input_folder}/*.hdf5')
    print(f'Found {len(input_files)} files.')
    if len(input_files) == 0:
        print(f'No files found. for pattern {input_folder}*.simtel.gz Aborting')
        return

    if os.path.exists(output_file):
        click.confirm('Output file exists. Overwrite?', abort=True)
        os.remove(output_file)

    n_chunks = (len(input_files) // chunk_size) + 1
    chunks = np.array_split(input_files, n_chunks)
    with Parallel(n_jobs=n_jobs, verbose=1) as parallel:
        for chunk in tqdm(chunks):
            results = parallel(delayed(read_file)(f) for f in chunk)

            runs = pd.concat([r[0] for r in results])
            array_events = pd.concat([r[1] for r in results])
            telescope_events = pd.concat([r[2] for r in results])

            fact.io.write_data(runs, output_file, key='runs', mode='a')
            fact.io.write_data(array_events, output_file, key='array_events', mode='a')
            fact.io.write_data(telescope_events, output_file, key='telescope_events', mode='a')

            if verify:
                verify_file(output_file)


def read_file(f):
    run = fact.io.read_data(f, key='runs')
    array_events = fact.io.read_data(f, key='array_events')
    telescope_events = fact.io.read_data(f, key='telescope_events')
    return run, array_events, telescope_events


def verify_file(input_file_path):
    runs = fact.io.read_data(input_file_path, key='runs')
    runs.set_index('run_id', drop=True, verify_integrity=True, inplace=True)

    telescope_events = fact.io.read_data(input_file_path, key='telescope_events')
    telescope_events.set_index(['run_id', 'array_event_id', 'telescope_id'], drop=True, verify_integrity=True, inplace=True)

    array_events = fact.io.read_data(input_file_path, key='array_events')
    array_events.set_index(['run_id', 'array_event_id'], drop=True, verify_integrity=True, inplace=True)



if __name__ == '__main__':
    main()
