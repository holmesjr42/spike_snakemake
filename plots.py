import nn_plots as nnp

# files = ['cr33_190321', 'cr50_190724', 'cr79_200211']
files = ['cr31_190312', 'cr33_190321', 'cr35_190403', 'cr50_190724', 'cr79_200211']

for animal in files:
    nnp.nn_3_1(animal)
    nnp.nn_1_1(animal)
