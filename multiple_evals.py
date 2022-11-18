import subprocess 

n_slices = 50 # number of timesteps on which to perform additive smoothing 
models = 'conjugate_forward conjugate_backward gru_backward' # name of the trained variational models to evaluate
exp_dirs = ['experiments/p_chaotic_rnn/2022_10_25__23_26_45'] # paths to the directories containing the trainings (multiple is possible)

processes = [subprocess.Popen(f'python eval.py --exp_dir {exp_dir} --n_slices {n_slices} --models {models}', shell=True) for exp_dir in exp_dirs]

for p in processes: 
    p.wait()

