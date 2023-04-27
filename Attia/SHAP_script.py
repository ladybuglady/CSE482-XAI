# Read these to understand how the model is saved and how you can load it:
# https://www.tensorflow.org/guide/keras/save_and_serialize
# https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk

# I have used the NEW KERAS SAVEDMODEL filesystem (updated January 2022).
# This is a directory that contains MODEL ARCHITECTURE as well as WEIGHTS.

# saved_model.pb IS THE MODEL ARCHITECTURE
# variables/variables.data-000000-of-00001 are the WEIGHTS.

# You will have to load the architecture then load it with the weights to initialize.


# the long string of numbers at the end of the directory (88......) is the test accuracy as
# a percent, missing the decimal point. Assume the decimal follows the first two digits (so 88.something %)
# This accuracy is obtained via only 3,000 recording files.

# The attia model reports 83% accuracy, so this should be sufficient to do a good SHAPLY analysis.

# good luck! -Zeynep