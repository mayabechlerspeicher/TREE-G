import datasets
import run_node_task_example as run
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
    parser.add_argument('--mask_type_probability', dest='attention_type_sample_probability', type=float, default='0.5',
                        help='Probability of sampling each mask type')
    args = parser.parse_args()
    exp_name = args.exp_name
    if exp_name == 'pubmed':
        dataset = datasets.Planetoid_PUBMED()
    elif exp_name == 'cora':
        dataset = datasets.Planetoid_CORA()
    elif exp_name == 'citeseet':
        dataset = datasets.Planetoid_CITESEER()
    elif exp_name == 'arxiv':
        dataset = datasets.OGB_ARXIV()
    else:
        raise 'dataset not found, options are: pubmed, cora, citeseet, arxiv'
    run.parallel_multiclass_train_val_test(dataset, with_constant_one_feature=False, max_attention_depth=args.a,
                                           max_walk_length=args.d, use_wandb=args.wandb, n_estimators=args.n_estimators,
                                           learning_rate=args.learning_rate,
                                           attention_type_sample_probability=args.attention_type_sample_probability)
