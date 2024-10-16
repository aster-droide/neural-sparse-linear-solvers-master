import os
import shutil

# Define source and destination directories
source_folder = 'C:\\Users\\astrid\Downloads\stand_small\stand_small_test'
destination_folder = 'C:\\Users\\astrid\Downloads\stand_small\\test1'
files = os.listdir(source_folder)
files.sort()

# move the first x files
x = 20
for file in files[:x]:
    source_file = os.path.join(source_folder, file)
    destination_file = os.path.join(destination_folder, file)
    if os.path.isfile(source_file):
        shutil.copy(source_file, destination_file)

print(f"First {x} files moved successfully.")
