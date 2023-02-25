import datasets
import run_node_task_example as run
import argparse

if __name__ == '__main__':
    with_constant_one_feature = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', dest='exp_name', type=str, default='pubmed')
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
    run.parallel_multiclass_train_val_test(dataset, with_constant_one_feature=False)
