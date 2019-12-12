# source: https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python

import os, sys

class HiddenPrints:
    '''
    Short class to hide the print of some code sections.

    ##############
    ### Usage: ###
    ##############

    with HiddenPrints():
        print("This will not be printed")
    print("This will be printed as before")

    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout