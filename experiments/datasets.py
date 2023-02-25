from torch_geometric.datasets import TUDataset, Planetoid, GitHub, KarateClub
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def OGB_MOLHIV():
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
    return dataset

def OGB_ARXIV():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    return dataset


def TU_MUTAG():
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    return dataset

def TU_NCI1():
    dataset = TUDataset(root='data/TUDataset', name='NCI1')
    return dataset


def TU_PTCMR():
    dataset = TUDataset(root='data/TUDataset', name='PTC_MR')
    return dataset


def TU_IMDBB():
    dataset = TUDataset(root='data/TUDataset', name='IMDB-BINARY')
    return dataset


def TU_IMDBM():
    dataset = TUDataset(root='data/TUDataset', name='IMDB-MULTI')
    return dataset


def TU_DD():
    dataset = TUDataset(root='data/TUDataset', name='DD')
    return dataset


def TU_PROTEINS():
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS_full')
    return dataset


def TU_ENZYMES():
    dataset = TUDataset(root='data/TUDataset', name='ENZYMES')
    return dataset


def TU_MUTAGANECY():
    dataset = TUDataset(root='data/TUDataset', name='Mutagenicity')
    return dataset


def Planetoid_CORA():
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    return dataset


def Planetoid_CITESEER():
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer')
    return dataset


def Planetoid_PUBMED():
    dataset = Planetoid(root='data/Planetoid', name='PubMed')
    return dataset
