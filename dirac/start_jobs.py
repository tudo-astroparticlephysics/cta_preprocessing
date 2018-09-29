import os
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac
from random import randint
import click
from tqdm import tqdm
from DIRAC.Core.Base import Script
Script.parseCommandLine()  # this might be needed. No idea why.


@click.command()
@click.argument('dataset')
@click.option('-d', '--delete', help='delete template Files', is_flag=True)
def main(dataset, delete):
    '''
    Dataset is the output of cta-prod3-dump-dataset as described here for example
    https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_MC_PROD3_Status

    '''
    dirac = Dirac()

    # info_zenith = {}
    # info_zenith["electron"] = {"20": "1154", "20_180": "1159", "40": "1189", "40_180": "1193"}
    # info_zenith["gamma"] = {"20": "1150", "20_180": "1157", "40": "1186", "40_180": "1191"}
    # info_zenith["gamma-diffuse"] = {"20": "1153", "20_180": "1158", "40": "1188", "40_180": "1192"}
    # info_zenith["proton"] = {"20": "1155", "20_180": "1161", "40": "1190", "40_180": "1194"}
    # print("dirac-dms-find-lfns Path=/vo.cta.in2p3.fr/MC/PROD3/LaPalma/" + art + "/simtel/" + info_zenith[art][zenith] + "/Data")
    with open(dataset) as f:
        simtel_files = f.readlines()

    print('Analysing {}'.format(len(lines)))

    server_liste = ["TORINO-USER", "CYF-STORM-USER", "CYF-STORM-Disk", "M3PEC-Disk", "OBSPM-Disk", "POLGRID-Disk", "FRASCATI-USER", "LAL-Disk", "CIEMAT-Disk", "CIEMAT-USER", "CPPM-Disk", "LAL-USER", "CYFRONET-Disk"]
    server_liste2 = ["DESY-ZN-USER", "M3PEC-USER", "LPNHE-Disk", "LPNHE-USER", "LAPP-USER", "LAPP-Disk"]

    # for i in server_liste3:
    #    for j in range(5):
    #        server_liste.append(i)
    for simtel_filename in tqdm(sorted(simtel_files)):
        file_basename = os.path.basename(simtel_filename)
        # pilot_filename = 'Templet_files/' + file_name + "_pilot.sh"
        # python_filename = 'Templet_files/' + file_name + "_preprocess.py"
        # c_filename = 'Templet_files/' + file_name + "_c.py"

        server_uploade = randint(0, len(server_liste) - 1)
        pilot_conf = open('temp_pilot.sh', 'r').read().replace("runid", file_name).replace('Number', info_zenith[art][zenith]).replace('uploade_server', server_liste[server_uploade]).replace('filename', file_name).replace('tel_type', tel_type).replace('camera_type', camera_type).replace("c.py", c_filename.replace("Templet_files/", "")).replace("preprocessing.py", python_filename.replace("Templet_files/", ""))
        pilot_out = open(pilot_filename, 'w')
        pilot_out.write(pilot_conf)
        pilot_out.close()

        python_conf = open('temp_preprocessing2.py', 'r').read().replace('Number', info_zenith[art][zenith]).replace('uploade_server', server_liste[server_uploade])
        python_out = open(python_filename, 'w')
        python_out.write(python_conf)
        python_out.close()

        c_conf = open('temp_c.py', 'r').read().replace('filename', file_name)
        c_out = open(c_filename, 'w')
        c_out.write(c_conf)
        c_out.close()


        input_sandbox = ["LFN:" + simtel_filename]
        j = Job()
        # Set Runtime 1h
        j.setCPUTime(60 * 60 * 12 * 8)
        j.setName('Cusotm process test')
        j.setInputSandbox(input_sandbox)
        j.setExecutable(pilot_filename.replace("Templet_files", "."))
        # This Server have mini conda Version installed
        j.setDestination(['LCG.IN2P3-CC.fr', 'LCG.DESY-ZEUTHEN.de', 'LCG.CNAF.it',
                          'LCG.GRIF.fr', 'LCG.CYFRONET.pl',
                          'LCG.Prague.cz', 'LCG.CIEMAT.es'])


        value = dirac.submit(j)['Value']
        print(file_name + '\tSubmission Result: {}'.format(value))
        if delete:
            os.system("rm " + pilot_filename)
            os.system("rm " + python_filename)
            os.system("rm " + c_filename)


if __name__ == '__main__':
    main()
