import os

if os.name == 'Linux' or os.name == 'posix':
    #print("{}".format(os.path.dirname(__file__)))
    #os.environ['LD_LIBRARY_PATH'] = os.getcwd()
    os.environ['LD_LIBRARY_PATH'] = os.path.dirname(__file__)
    #print("$LD_LIBRARY_PATH: {}".format(os.environ['LD_LIBRARY_PATH']))
