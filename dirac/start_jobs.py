import os
from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac
from random import randint
import click
from tqdm import tqdm


def delete_empty(array, check=None):
    array_new = []
    for data in array:
        if data != "":
            if check != None:
                if check in data:
                    array_new.append(data)
            else:
                array_new.append(data)
    return array_new


@click.command()
@click.argument('tel_type', default='all')  # , help='LST or MST or SST'
@click.argument('camera_type', default='all')  # , help='LST: LSTCam, MST: NectarCam, FlashCam, SCTCam, SST: CHEC'
@click.argument('art', default='gamma')  # , help='gamma, gamma-diffuse, proton, electron'
@click.argument('zenith', default='20')  # , help='20, 40'
@click.option('-d', '--delete', help='delete template Files', is_flag=True)
def main(tel_type, camera_type, art, zenith, delete):
    print(tel_type)
    print(camera_type)
    print(art)
    dirac = Dirac()

    info_zenith = {}
    info_zenith["electron"] = {"20": "1154", "20_180": "1159", "40": "1189", "40_180": "1193"}
    info_zenith["gamma"] = {"20": "1150", "20_180": "1157", "40": "1186", "40_180": "1191"}
    info_zenith["gamma-diffuse"] = {"20": "1153", "20_180": "1158", "40": "1188", "40_180": "1192"}
    info_zenith["proton"] = {"20": "1155", "20_180": "1161", "40": "1190", "40_180": "1194"}
    print("dirac-dms-find-lfns Path=/vo.cta.in2p3.fr/MC/PROD3/LaPalma/" + art + "/simtel/" + info_zenith[art][zenith] + "/Data")

    files = delete_empty(os.popen("dirac-dms-find-lfns Path=/vo.cta.in2p3.fr/MC/PROD3/LaPalma/" + art + "/simtel/" + info_zenith[art][zenith] + "/Data").read().split("\n"), "/vo.cta.in2p3.fr/MC/PROD3")
    print(len(files))
    conf = {"1150": "gamma_20deg_0deg_run{}___cta-prod3-lapalma3-2147m-LaPalma.simtel.gz", "1157": "gamma_20deg_180deg_run{}___cta-prod3-lapalma3-2147m-LaPalma.simtel.gz", "1186": "gamma_40deg_0deg_run{}___cta-prod3-lapalma3-2147m-LaPalma.simtel.gz", "1191": "gamma_40deg_180deg_run{}___cta-prod3-lapalma3-2147m-LaPalma.simtel.gz"}
    os.system("mkdir -p Templet_files")

    server_liste = ["TORINO-USER", "CYF-STORM-USER", "CYF-STORM-Disk", "M3PEC-Disk", "OBSPM-Disk", "POLGRID-Disk", "FRASCATI-USER", "LAL-Disk", "CIEMAT-Disk", "CIEMAT-USER", "CPPM-Disk", "LAL-USER", "CYFRONET-Disk"]
    server_liste2 = ["DESY-ZN-USER", "M3PEC-USER", "LPNHE-Disk", "LPNHE-USER", "LAPP-USER", "LAPP-Disk"]
    # server_liste3 = ["CC-IN2P3-USER"]  # this Server have Problems
    for i in server_liste2:
        for j in range(3):
            server_liste.append(i)

    # for i in server_liste3:
    #    for j in range(5):
    #        server_liste.append(i)
    erste = 0
    anzahl_files = 0
    for simtel_filename in tqdm(sorted(files)):
        anzahl_files += 1
        if anzahl_files < 20:
            continue
        if anzahl_files == 41:
            break
        file_name = simtel_filename.split("/")
        file_name = file_name[len(file_name) - 1]
        file_name2 = file_name.split("___")[0]
        pilot_filename = 'Templet_files/' + file_name + "_pilot.sh"
        python_filename = 'Templet_files/' + file_name + "_preprocess.py"
        c_filename = 'Templet_files/' + file_name + "_c.py"

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

        input_sandbox = [pilot_filename, python_filename, c_filename,
                         "LFN:/vo.cta.in2p3.fr/user/t/thomas.jung/tar/ctapipe2.tar.xz", "LFN:/vo.cta.in2p3.fr/user/t/thomas.jung/tar/pyhessio.tar.xz", "LFN:/vo.cta.in2p3.fr/user/t/thomas.jung/tar/ctapipe-extra.tar.xz", "LFN:/vo.cta.in2p3.fr/user/t/thomas.jung/tar/neighbor.p", "LFN:/vo.cta.in2p3.fr/user/t/thomas.jung/tar/leakage.p"
                         ]
        input_sandbox.append("LFN:" + simtel_filename)
        job_name = file_name + "_" + str(tel_type) + "_" + str(camera_type)
        j = Job()
        # Set Runtime 1h
        j.setCPUTime(60 * 60 * 12 * 8)
        j.setName(job_name)
        j.setInputSandbox(input_sandbox)
        j.setExecutable(pilot_filename.replace("Templet_files", "."))
        # This Server have mini conda Version installed
        j.setDestination(['LCG.IN2P3-CC.fr', 'LCG.DESY-ZEUTHEN.de', 'LCG.CNAF.it',
                          'LCG.GRIF.fr', 'LCG.CYFRONET.pl',
                          'LCG.Prague.cz', 'LCG.CIEMAT.es'])
        # This Server have mini conda Version installed and disk Quota which is very small
        # , 'LCG.LAPP.fr'
        # , 'LCG.PRAGUE-CESNET.cz'
        value = dirac.submit(j)['Value']
        #print(file_name + '\tSubmission Result: {}'.format(value))
        if delete:
            os.system("rm " + pilot_filename)
            os.system("rm " + python_filename)
            os.system("rm " + c_filename)


if __name__ == '__main__':
    main()
