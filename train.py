import jax 
import jax.numpy as jnp
from backward_ica import variational

import backward_ica.utils as utils
import backward_ica.stats.hmm as hmm
import backward_ica.variational as variational
import backward_ica.stats as stats
from backward_ica.training import SVITrainer, define_frozen_tree


    
def main(args):


    if args.float64: 
        utils.enable_x64(True)
    stats.set_parametrization(args)

    p = hmm.get_generative_model(utils.load_args('args', args.exp_dir))
    theta_star = utils.load_params('theta_star', args.exp_dir)
    data = jnp.load(os.path.join(args.exp_dir, 'obs_seqs.npy'))


    key_phi = jax.random.PRNGKey(args.seed)

    
    q = variational.get_variational_model(args, 
                                        p=p)


    frozen_params = define_frozen_tree(key_phi, 
                                        args.frozen_params, 
                                        q, 
                                        theta_star)


    trainer = SVITrainer(p=p, 
                        theta_star=theta_star,
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        num_samples=args.num_samples,
                        force_full_mc=args.full_mc,
                        frozen_params=frozen_params,
                        sweep_sequences=args.sweep_sequences)


    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)

    params = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                            data=data, 
                                                            num_fits=args.num_fits,
                                                            log_dir=args.save_dir,
                                                            store_every=args.store_every,
                                                            args=args)[0] # returns the best fit (based on the last value of the elbo)
    
    # utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), args.save_dir, plot=True, best_epochs_only=True)
    utils.save_params(params, 'phi', args.save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime

    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--exp_dir', type=str, default='experiments/p_linear/2022_11_03__16_07_11')

    parser.add_argument('--sweep_sequences', action='store_true')
    parser.add_argument('--num_fits', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=1)
    
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--store_every', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    args.full_mc = 'full_mc' in args.model # whether to force the use the full MCMC ELBO (e.g. prevent using closed-form terms even with linear models)
    args.frozen_params  = args.model.split('__')[1:] # list of parameter groups which are not learnt
    args.save_dir = os.path.join(args.exp_dir, args.model)
    os.makedirs(args.save_dir, exist_ok=True)

    
    args = utils.get_defaults(args)

    utils.save_args(args, 'args', args.save_dir)
    args_p = utils.load_args('args', args.exp_dir)
    args.state_dim, args.obs_dim = args_p.state_dim, args_p.obs_dim
    main(args)

