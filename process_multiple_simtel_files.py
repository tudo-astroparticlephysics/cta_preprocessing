import click
import numpy as np
from tqdm import tqdm
import os
import glob
from joblib import delayed, Parallel
from process_simtel_file import process_file, write_result_to_file
from preprocessing.parameters import PREPConfig


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_folder', type=click.Path(dir_okay=True,
                                                 file_okay=False))
@click.argument('config_file',
                type=click.Path(file_okay=True)
                )
def main(input_pattern, output_folder, config_file):
    '''
    process simtel files given as matching INPUT_PATTERN into several
    hdf5 files saved in OUTPUT_FOLDER
    with the same filename as the input but with .h5 extension.
    '''
    config = PREPConfig(config_file)
    input_files = glob.glob(input_pattern)
    print(f'Found {len(input_files)} files matching pattern.')

    if len(input_files) == 0:
        print(f'No files found. For pattern {input_pattern}. Aborting')
        return

    def output_file_for_input_file(input_file):
        return os.path.join(output_folder,
                            os.path.basename(input_file).replace('simtel.gz',
                                                                 'h5'))

    if not config.overwrite:
        input_files = list(filter(lambda v: not os.path.exists(output_file_for_input_file(v)),
                                  input_files))
        print(f'''Preprocessing on {len(input_files)}
              files that have no matching output''')
    else:
        print('Preprocessing all found input_files'
              ' and overwriting existing output.')
        output_files = [output_file_for_input_file(f) for f in input_files]
        [os.remove(of) for of in output_files if os.path.exists(of)]

    if len(input_files) < 1:
        print('No files to process')
        return

    n_chunks = (len(input_files) // config.chunksize) + 1
    chunks = np.array_split(input_files, n_chunks)

    with Parallel(n_jobs=config.n_jobs, verbose=config.verbose) as parallel:
        for chunk in tqdm(chunks):
            results = parallel(delayed(process_file)(f, config) for f in chunk)
            for input_file, r in zip(input_files, results):
                # from IPython import embed; embed()
                run_info_container, array_events, telescope_events = r
                output_file = output_file_for_input_file(input_file)
                write_result_to_file(run_info_container,
                                     array_events,
                                     telescope_events,
                                     output_file)
                print(f'processed file {input_file}, writing to {output_file}')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
