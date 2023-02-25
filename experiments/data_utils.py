import numpy as np

def add_constant_one_feature(Xs):
    for x in Xs:
        ones_vec = np.ones(shape=(x.get_number_of_nodes(), 1))
        x.features = np.append(x.features, ones_vec, axis=1)


def print_trees(model):
    for idx, estimator in enumerate(model.estimators_):
        print('Tree ' + str(idx) + ':')
        estimator[0].print_tree()
