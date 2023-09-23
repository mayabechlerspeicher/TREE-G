import datasets
import run_graph_task_example as run
import argparse

if __name__ == '__main__':
    with_constant_one_feature = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', dest='exp_name', type=str, default='pubmed')
    parser.add_argument('--a', dest='a', type=int, default='3',
                        help='The maximal distance of the considered subsets from the target node')
    parser.add_argument('--d', dest='d', type=int, default='3', help='Maximal walk length')
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='Log to wandb')
    parser.add_argument('--n_estimators', dest='n_estimators', type=int, default='50', help='Number of trees')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default='0.1', help='Learning rate')
    parser.add_argument('--mask_type_probability', dest='mask_type_probability', type=float, default='0.25',
                        help='Probability of sampling each mask type')
    args = parser.parse_args()
    exp_name = args.exp_name
    if exp_name == 'hiv':
        dataset = datasets.OGB_MOLHIV
        run.multiclass_train_val_test_ogb_splits(dataset=dataset, with_constant_one_feature=True,
                                                 max_attention_depth=args.a, max_walk_length=args.d, use_wandb=args.wandb,
                                                 n_estimators=args.n_estimators,
                                                 learning_rate=args.learning_rate,
                                                 attention_type_sample_probability=args.mask_type_probability)
    else:
        if exp_name == 'proteins':
            dataset = datasets.TU_PROTEINS()
        elif exp_name == 'mutag':
            dataset = datasets.TU_MUTAG()
        elif exp_name == 'dd':
            dataset = datasets.TU_DD()
        elif exp_name == 'enzymes':
            datasets = datasets.TU_ENZYMES()
        elif exp_name == 'nci1':
            datasets = datasets.TU_NCI1()
        elif exp_name == 'imdbb':
            datasets = datasets.TU_IMDBB()
        elif exp_name == 'imdbm':
            datasets = datasets.TU_IMDBM()
        elif exp_name == 'ptcmr':
            datasets = datasets.TU_PTCMR()
        elif exp_name == 'mutagenicity':
            datasets = datasets.TU_MUTAGANECY()
        else:
            raise 'dataset not found, the options are: proteins, mutag, dd, enzymes, nci1, imdbb, imdbm, ptcmr, mutagenicity, hiv'

        run.parallel_multiclass_cross_validation(dataset=dataset, with_constant_one_feature=False,
                                                 max_attention_depth=args.a, max_walk_length=args.d, use_wandb=args.wandb,
                                                 n_estimators=args.n_estimators,
                                                 learning_rate=args.learning_rate,
                                                 attention_type_sample_probability=args.mask_type_probability)
