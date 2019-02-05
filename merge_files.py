import fact.io
import click
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from process_simtel_file import write_result_to_file
from colorama import Fore, Style

@click.command()
@click.argument(
    'input_pattern', type=str
)
@click.argument(
    'output_file', type=click.Path(
        dir_okay=False,
        file_okay=True,
))
@click.option('--verify/--no-verify', default=False, help='Wether to verify the output file ')
@click.option('-j', '--n_jobs', default=1, help='number of jobs to use for reading data')
@click.option('-c', '--chunk_size', default=50, help='files per chunk')
def main(input_pattern, output_file, verify, n_jobs, chunk_size):
    """
    Process multiple simtel files gievn as INPUT_FILES into one hdf5 file saved in OUTPUT_FILE.
    The hdf5 file will contain three groups. 'runs', 'array_events', 'telescope_events'.

    These files can be put into the classifier tools for learning.
    See https://github.com/fact-project/classifier-tools
    """

    input_files = glob.glob(input_pattern)
    print(f'Found {len(input_files)} files.')
    if len(input_files) == 0:
        print(f'No files found. for pattern {input_folder}*.simtel.gz Aborting')
        return

    if os.path.exists(output_file):
        click.confirm('Output file exists. Overwrite?', abort=True)
        os.remove(output_file)

    n_chunks = (len(input_files) // chunk_size) + 1
    chunks = np.array_split(input_files, n_chunks)

    with pd.HDFStore(output_file) as hdf_store:

        with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
            for chunk in tqdm(chunks):
                results = parallel(delayed(read_file)(f) for f in chunk)

                runs = pd.concat([r[0] for r in results])
                array_events = pd.concat([r[1] for r in results])
                telescope_events = pd.concat([r[2] for r in results])


                hdf_store.append('runs', runs)
                hdf_store.append('array_events', array_events)
                hdf_store.append('telescope_events', telescope_events)

    if verify:
        verify_file(output_file)


def read_file(f):
    telescope_events = pd.read_hdf(f, 'telescope_events')
    array_events = pd.read_hdf(f, 'array_events')
    run = pd.read_hdf(f, 'runs')
    return run, array_events, telescope_events


def verify_file(input_file_path):
    try:
        telescope_events = pd.read_hdf(input_file_path, 'telescope_events')
        array_events = pd.read_hdf(input_file_path, 'array_events')
        runs = pd.read_hdf(input_file_path, 'runs')

        runs.set_index('run_id', drop=True, verify_integrity=True, inplace=True)
        telescope_events.set_index(['run_id', 'array_event_id', 'telescope_id'], drop=True, verify_integrity=True, inplace=True)
        array_events.set_index(['run_id', 'array_event_id'], drop=True, verify_integrity=True, inplace=True)

        print(Fore.GREEN + Style.BRIGHT + f'File {input_file_path} seems fine.   ✔ ')
    except:
        print(Fore.RED + f'File {input_file_path} seems to be broken.')



if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
