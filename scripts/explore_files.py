import numpy as np

file_1 = np.load("C:\\Users\\astrid\Downloads\stand_small\\test\\1833244380076257152_0bed54b8934055d852611e2226b9bedb11dc3d20_953ae49cadb590e92658716a09d2952dbde4bdaf.npz")
file_2 = np.load("C:\\Users\\astrid\Downloads\stand_small\\test\\1805599623988272222_1af7a69cf1ccc66a650d4c7445977004e17c4c1d_f54177ce98988a461993b96a350deded203acc83.npz")

# display contents of each file
file_1_contents = {key: file_1[key] for key in file_1}
file_2_contents = {key: file_2[key] for key in file_2}

print(file_1_contents)
print(file_2_contents)