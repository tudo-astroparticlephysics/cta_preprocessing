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
@click.option('-f', '--hdf_format', default='tables', type=click.Choice(['tables', 'h5py']))
def main(input_pattern, output_file, verify, n_jobs, chunk_size, hdf_format):
    """
    Merge multiple hdf5 files matched by INPUT_PATTERN into one hdf5 file saved in OUTPUT_FILE.
    The hdf5 file will contain three groups. 'runs', 'array_events', 'telescope_events'.

    These files can be put into the classifier tools for learning.
    See https://github.com/fact-project/classifier-tools
    """

    input_files = glob.glob(input_pattern)
    print(f'Found {len(input_files)} files.')
    if len(input_files) == 0:
        print(f'No files found for pattern {input_pattern} Aborting')
        return

    if os.path.exists(output_file):
        click.confirm('Output file exists. Overwrite?', abort=True)
        os.remove(output_file)

    n_chunks = (len(input_files) // chunk_size) + 1
    chunks = np.array_split(input_files, n_chunks)

    if hdf_format == 'tables':
        with pd.HDFStore(output_file) as hdf_store:
            for chunk in tqdm(chunks):
                results = [read_file(f) for f in chunk]
                runs = pd.concat([r[0] for r in results])
                array_events = pd.concat([r[1] for r in results])
                telescope_events = pd.concat([r[2] for r in results])

                sort_arrays_inplace(runs, array_events, telescope_events)

                hdf_store.append('runs', runs)
                hdf_store.append('array_events', array_events)
                hdf_store.append('telescope_events', telescope_events)

    else:
        
        import fact.io
        for chunk in tqdm(chunks):
            results = [read_file(f) for f in chunk]

            runs = pd.concat([r[0] for r in results])
            array_events = pd.concat([r[1] for r in results])
            telescope_events = pd.concat([r[2] for r in results])
            
            sort_arrays_inplace(runs, array_events, telescope_events)

            fact.io.write_data(runs, output_file, key='runs', mode='a')
            fact.io.write_data(array_events, output_file, key='array_events', mode='a')
            fact.io.write_data(telescope_events, output_file, key='telescope_events', mode='a')


    if verify:
        verify_file(output_file, hdf_format)


def sort_arrays_inplace(runs, array_events, telescope_events):
    telescope_events.sort_values(by=['run_id', 'array_event_id', 'telescope_id'], inplace=True)
    array_events.sort_values(by=['run_id', 'array_event_id'], inplace=True)
    runs.sort_values(by=['run_id'], inplace=True)


def read_file(f):
    telescope_events = pd.read_hdf(f, 'telescope_events')
    array_events = pd.read_hdf(f, 'array_events')
    run = pd.read_hdf(f, 'runs')
    return run, array_events, telescope_events


def verify_file(input_file_path, format='tables'):
    try:
        if format == 'tables':
            telescope_events = pd.read_hdf(input_file_path, 'telescope_events')
            array_events = pd.read_hdf(input_file_path, 'array_events')
            runs = pd.read_hdf(input_file_path, 'runs')
        else:
            import fact.io
            telescope_events = fact.io.read_data(input_file_path, key='telescope_events')
            array_events = fact.io.read_data(input_file_path, key='array_events')
            runs = fact.io.read_data(input_file_path, key='runs')

        runs.set_index('run_id', drop=True, verify_integrity=True, inplace=True)
        telescope_events.set_index(['run_id', 'array_event_id', 'telescope_id'], drop=True, verify_integrity=True, inplace=True)
        array_events.set_index(['run_id', 'array_event_id'], drop=True, verify_integrity=True, inplace=True)
        
        print(Fore.GREEN + Style.BRIGHT + f'File "{input_file_path}" seems fine.   ✔ ')
        print(Style.RESET_ALL)   
    except:
        print(Fore.RED + f'File {input_file_path} seems to be broken. \n')
        print(Style.RESET_ALL)   
        import sys, traceback
        traceback.print_exc(file=sys.stdout)



if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
