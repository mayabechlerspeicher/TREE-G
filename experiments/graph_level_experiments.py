import datasets
import run_graph_task_example as run
import argparse


if __name__ == '__main__':
    with_constant_one_feature = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', dest='exp_name', type=str, default='pubmed')
    args = parser.parse_args()
    exp_name = args.exp_name
    if exp_name == 'hiv':
        dataset = datasets.OGB_MOLHIV
        run.multiclass_train_val_test_ogb_splits(dataset=dataset, with_constant_one_feature=True)
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

        run.parallel_multiclass_cross_validation(dataset=dataset, with_constant_one_feature=False)
