import os
import glob
import click
import numpy as np
from tqdm import tqdm
from joblib import delayed, Parallel
from process_simtel_file import process_file, write_result_to_file
from preprocessing.parameters import PREPConfig
import logging
from logging.config import dictConfig
import yaml


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_folder', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('config_file', type=click.Path(file_okay=True))
@click.option(
    '-l',
    '--logger_config_file',
    default='logging_config.yaml',
    help='Specify a yaml logger config file to tune console and file debugging',
)
@click.option('-n', '--n_events', default=-1, help='Number of events to process in each file.')
@click.option(
    '-j',
    '--n_jobs',
    default=1,
    help='Number of jobs to start.' 'This is usefull when passing more than one simtel file.',
)
@click.option(
    '--overwrite/--no-overwrite',
    default=False,
    help='If false (default) will only process non-existing filenames',
)
@click.option('-v', '--verbose', default=1, help='specifies the output being shown during processing')
@click.option('-c', '--chunksize', default=1, help='number of files per chunk')
def main(
    input_pattern,
    output_folder,
    config_file,
    logger_config_file,
    n_events,
    n_jobs,
    overwrite,
    verbose,
    chunksize,
):
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
    # workaround https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    try:
        with open(logger_config_file, 'rb') as f:
            config = yaml.safe_load(f)
        dictConfig(config)
    except Exception:
        logging.warning('Could not set logger configuration.', exc_info=True)

    config = PREPConfig(config_file)

    if not input_pattern.endswith('simtel.gz'):
        logging.warning(
            'WARNING. Pattern does not end with file extension (simtel.gz). More files might be matched.'
        )

    input_files = glob.glob(input_pattern)

    if len(input_files) == 0:
        logging.critical(f'No files found. For pattern {input_pattern}. Aborting')
        return
    else:
        logging.info(f'Found {len(input_files)} files matching pattern.')

    def output_file_for_input_file(input_file):
        output_file = os.path.join(output_folder, os.path.basename(input_file).replace('simtel.gz', 'h5'))
        return output_file

    if not overwrite:
        input_files = list(filter(lambda v: not os.path.exists(output_file_for_input_file(v)), input_files))
        logging.info(f'Preprocessing on {len(input_files)} files that have no matching output')
    else:
        output_files = [output_file_for_input_file(f) for f in input_files]
        [os.remove(of) for of in output_files if os.path.exists(of)]
        logging.info('Preprocessing all found input_files and overwriting existing output.')

    n_chunks = (len(input_files) // chunksize) + 1
    chunks = np.array_split(input_files, n_chunks)
    logging.debug(f'Splitted input_files in {n_chunks} chunks')

    with Parallel(n_jobs=n_jobs, verbose=verbose, backend='loky') as parallel:
        for chunk in tqdm(chunks):
            results = parallel(
                delayed(process_file)(f, config, n_jobs=1, n_events=n_events, verbose=verbose) for f in chunk
            )  # 1 because multiple threads on one file did not perform well at all
            if len(results) != len(chunk):
                logging.error('One or more files failed to process in this chunk.')

            assert len(results) == len(chunk)

            for input_file, r in zip(chunk, results):
                if r:
                    run_info_container, array_events, telescope_events = r
                    output_file = output_file_for_input_file(input_file)
                    write_result_to_file(run_info_container, array_events, telescope_events, output_file)
                else:
                    logging.error(f'could not process file {input_file}. job did not return a result')


if __name__ == '__main__':
    main()
