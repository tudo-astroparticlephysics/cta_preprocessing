import datetime

def create_logfile(path):
    with open(path, 'w') as logfile:
        logfile.write('Starting run at' + str(datetime.datetime.now()))

def write_to_log(path, message):
    with open(path, 'a') as logfile:
        logfile.write('\n'+message)