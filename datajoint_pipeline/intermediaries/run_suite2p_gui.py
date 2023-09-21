import suite2p
from suite2p.gui import gui2p
import sys
import getopt


def main(file_name):
    # print(file_name)
    gui2p.run(file_name)


if __name__ == "__main__":
    # The first argument back from the command line is the name of the script 'si_header_cmd.py', so we can skip this.
    argv = sys.argv[1:]
    # print(argv)
    # Fetch the arguments given from the command line, allowing the option -i for the input file.
    ops, args = getopt.getopt(argv, "i:", ["ifile="])
    # print(ops)
    # The output of ops is a list in a list - access the first item to get the first option file pair and check that
    # it is using the correct tag.
    if ops[0][0] == '-i':
        # print(ops[0][0])
        main(ops[0][1])