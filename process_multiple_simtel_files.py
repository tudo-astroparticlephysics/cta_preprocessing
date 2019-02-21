import os
import glob

import click
import numpy as np
from tqdm import tqdm
from joblib import delayed, Parallel
from process_simtel_file import process_file, write_result_to_file
from preprocessing.parameters import PREPConfig
from colorama import Fore, Style


@click.command()
@click.argument(
    'input_pattern',
    type=str)
@click.argument(
    'output_folder',
    type=click.Path(dir_okay=True, file_okay=False))
@click.argument(
    'config_file',
    type=click.Path(file_okay=True))
@click.option(
    '-n', '--n_events',
    default=-1,
    help='Number of events to process in each file.')
@click.option(
    '-j', '--n_jobs',
    default=1,
    help='Number of jobs to start.'
    'This is usefull when passing more than one simtel file.')
@click.option(
    '--overwrite/--no-overwrite',
    default=False,
    help='If false (default) will only process non-existing filenames')
@click.option(
    '-v', '--verbose',
    default=1,
    help='specifies the output being shown during processing')
@click.option(
    '-c', '--chunksize',
    default=1,
    help='number of files per chunk')
def main(
        input_pattern, output_folder, config_file,
        n_events, n_jobs, overwrite, verbose, chunksize):
    '''
    Process simtel files given as matching
    'input_pattern'
    into one hdf5 file for each simtel file.
    Output files get placed into
    'output_folder'
    with the same filename as their respective input file but the
    extension switched to .hdf5

    Processing steps consist of:
    - Calibration
    - Calculating image features
    - Collecting MC header information
    The hdf5 file will contain three groups.
    'runs', 'array_events', 'telescope_events'.

    The config specifies which
    - telescopes
    - integrator
    - cleaning
    - cleaning levels per telescope type
    to use.
    '''
    config = PREPConfig(config_file)

    if not input_pattern.endswith('simtel.gz'):
        print(Fore.RED + Style.BRIGHT
              + f'''WARNING.
              Pattern does not end with file extension (simtel.gz).
              More files might be matched.''')
        print(Style.RESET_ALL)

    input_files = glob.glob(input_pattern)
    print(f'Found {len(input_files)} files matching pattern.')

    if len(input_files) == 0:
        print(f'No files found. For pattern {input_pattern}. Aborting')
        return

    def output_file_for_input_file(input_file):
        output_file = os.path.join(
            output_folder,
            os.path.basename(input_file).replace('simtel.gz', 'h5'))

        return output_file

    if not overwrite:
        input_files = list(filter(
            lambda v: not os.path.exists(
                output_file_for_input_file(v)), input_files))
        print(f'''Preprocessing on {len(input_files)}
              files that have no matching output''')
    else:
        output_files = [output_file_for_input_file(f) for f in input_files]
        [os.remove(of) for of in output_files if os.path.exists(of)]
        print('Preprocessing all found input_files'
              ' and overwriting existing output.')

    if len(input_files) < 1:
        print('No files to process')
        return

    n_chunks = (len(input_files) // chunksize) + 1
    chunks = np.array_split(input_files, n_chunks)

    with Parallel(
            n_jobs=n_jobs, verbose=verbose,
            backend='multiprocessing') as parallel:
        for chunk in tqdm(chunks):
            results = parallel(delayed(process_file)(
                f, config, n_jobs=1, n_events=n_events, verbose=verbose)
                for f in chunk)  # 1 because kai
            if len(results) != len(chunk):
                print(Fore.RED + Style.BRIGHT
                      + 'WARNING.'
                      'One or more files failed to process in this chunk.')

            assert len(results) == len(chunk)

            for input_file, r in zip(chunk, results):
                run_info_container, array_events, telescope_events = r
                output_file = output_file_for_input_file(input_file)
                print(f'processed file {input_file}, writing to {output_file}')
                write_result_to_file(
                    run_info_container, array_events,
                    telescope_events, output_file)


if __name__ == '__main__':
    main()
