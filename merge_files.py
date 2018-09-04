import fact.io
import click
import os
import glob
from tqdm import tqdm

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
def main(input_folder, output_file, verify):
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


    for f in tqdm(input_files):

        runs = fact.io.read_data(f, key='runs')
        array_events = fact.io.read_data(f, key='array_events')
        telescope_events = fact.io.read_data(f, key='telescope_events')

        fact.io.write_data(runs, output_file, key='runs', mode='a')
        fact.io.write_data(array_events, output_file, key='array_events', mode='a')
        fact.io.write_data(telescope_events, output_file, key='telescope_events', mode='a')

        if verify:
            verify_file(output_file)


def verify_file(input_file_path):
    runs = fact.io.read_data(input_file_path, key='runs')
    runs.set_index('run_id', drop=True, verify_integrity=True, inplace=True)

    telescope_events = fact.io.read_data(input_file_path, key='telescope_events')
    telescope_events.set_index(['run_id', 'array_event_id', 'telescope_id'], drop=True, verify_integrity=True, inplace=True)

    array_events = fact.io.read_data(input_file_path, key='array_events')
    array_events.set_index(['run_id', 'array_event_id'], drop=True, verify_integrity=True, inplace=True)



if __name__ == '__main__':
    main()
