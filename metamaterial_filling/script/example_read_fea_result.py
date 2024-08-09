 ## Use the following in your script to load the object from the file
import pickle
import argparse

from io_interface.fea_result_class import FEA_Opt_Result

argparser = argparse.ArgumentParser()
argparser.add_argument('-f', '--filepath', help='PKL filepath to save the result object.', type=str)

# Deserialize the object from the file
args = argparser.parse_args()
with open(args.filepath, 'rb') as file:
    loaded_obj = pickle.load(file)

# Show the result
loaded_obj.show_result()