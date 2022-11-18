# Codebase for the paper "Amortized backward variational inference for nonlinear state-space models"

## Installation 

1. Create an environment:
```shell 
python3 -m venv <path-to-your-new-env>
source <path-to-your-new-env>/bin/activate
pip install --upgrade pip
``` 
2. Install JAX
```shell
pip install --upgrade "jax[cpu]"
```

*In case of problems here, refer to [JAX installation instructions](https://github.com/google/jax#installation) for more informations.*

3. Install remaining dependencies: 

```shell 
pip install -r requirements.txt
```

## Use 

### Internals
All ingredients to build the training / inference routines are located in the folder [backward_ica](backward_ica/). In particular: 

- [the HMM classes](backward_ica/stats/hmm.py) used as generative models and the different mappings involved (linear / nonlinear).
- [the variational models](backward_ica/variational/models.py) with [the DNNs involved](backward_ica/variational/inference_nets.py).
- [the optimization routines](backward_ica/training.py).
- [the ELBO definitions](backward_ica/elbos.py).

### Training 

To train models use [this script](/train_multiple_models.py) following the argument definitions given as code comments. This will call the [data generation script](/generate_data.py) script to generate some common data for all models under some given $p^\theta$, then the [training script](/train.py) for each variational model separately. 

Everything ends up in `experiments/<name-of-the-generative-models>/<date>` with separate subfolders for the different variational models. 


### Evaluation 

To evaluate trained models on some data, use [this script](/multiple_evals.py) following the argment definitions given as code comments. This will generate a subfolder of the form ``experiments/<name-of-the-generative-models>/<date>/evals` containing the evaluation results of each different model. 

To plot the results and overlay multiple model evaluations, use [this script](/combine_evals.py).



