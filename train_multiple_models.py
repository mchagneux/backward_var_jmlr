import subprocess
import os
from datetime import datetime 

p_model = 'p_linear_transition_with_nonlinear_emission' # name of the generative model, 
                                                        # one of ['linear', 'linear_transition_with_nonlinear_emission', 'choatic_rnn']
base_dir = os.path.join('experiments', f'p_{p_model}')

q_models = ['conjugate_forward', 'conjugate_backward', 'gru_backward'] # name of the variational models to be trained
                                                                       # one of ['linear', 'gru_backward', 'conjugate_backward', 'conjugate_forward']

num_epochs = 200 # number of sweeps through the entire dataset
learning_rate = 0.01 
dims = '5 5' # model dimensonal in the format 'state_dim obs_dim'
batch_size = 100 # number of sequences in the minibatch for the stochastic gradient
num_seqs = 1000 # number of sequences in the dataset 
seq_length = 500 # length of each individual sequence
num_samples_list = [100] # number of monte carlo samples for the ELBO
load_from = '' # folder from which to load pre-generated data and / or parameters (if applicable, .e.g. chaotic_rnn).
loaded_seq = False # whether sequences are loaded from the 'load_from' argument
sweep_sequences = False # whether to take a gradient step after each new obseration (the ELBO an its gradient are recomputed for all subsequences)
store_every = 5 # for every num_epochs // store_every, the parameters will be stored 

os.makedirs(base_dir, exist_ok=True)


date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

loaded_seq = '--loaded_seq' if loaded_seq else ''
load_from = f'--load_from {load_from}' if load_from != '' else ''
sweep_sequences = f'--sweep_sequences' if sweep_sequences else ''

exp_dir = os.path.join(base_dir, date)
os.makedirs(exp_dir, exist_ok=True)

subprocess.run(f'python generate_data.py {loaded_seq} {load_from} \
                        --model {p_model} \
                        --dims {dims} \
                        --num_seqs {num_seqs} \
                        --seq_length {seq_length} \
                        --exp_dir {exp_dir}',  
                shell=True)


processes = [subprocess.Popen(f'python train.py {sweep_sequences} \
                                    --model {model} \
                                    --exp_dir {exp_dir} \
                                    --batch_size {batch_size} \
                                    --learning_rate {learning_rate} \
                                    --num_epochs {num_epochs} \
                                    --store_every {store_every} \
                                    --num_samples {num_samples}', 
                        shell=True) for model, num_samples in zip(q_models, num_samples_list)]

         
tensorboard_process = subprocess.Popen(f'tensorboard --logdir {exp_dir}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()
