import os, sys, time, random, argparse

from nas_201_api  import NASBench201API as API
from models       import CellStructure, get_search_spaces


def config2structure_func(max_nodes):
    def config2structure(config):
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = config[node_str]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return CellStructure( genotypes )
    return config2structure



class NAS201Benchmark(object):

    def __init__(self, dataset,
                arch_nas_dataset='../unified-hpo/lookup/NAS-Bench-201-v1_1-096897.pth'):
        self.max_nodes = 4
        if dataset == 'cifar10':
            dataset = 'cifar10-valid'
        self.dataset = dataset
        self.name = "nas-bench-201"
        if not os.path.isfile(arch_nas_dataset):
            raise ValueError("No lookup data available")
        
        self.api = API(arch_nas_dataset)
        self.search_space = get_search_spaces('cell', "nas-bench-201")
        self.convert_func = config2structure_func(self.max_nodes)

    def get_search_space(self):
        return self.search_space

    def convert_structure(self, config):
        return self.convert_func(config)

    def get_arch_index(self, config):
        structure = self.convert_structure(config)
        return self.api.query_index_by_arch(structure)

    def get_eval_result(self, arch_index, n_epochs=None):
        if n_epochs != None and n_epochs > 0:
            i_epochs = n_epochs - 1
        else:
            i_epochs = n_epochs

        info = self.api.get_more_info(arch_index, self.dataset, i_epochs, 
                                     hp='200', is_random=True)
        return info

    def query_by_arch(self, arch):
        return self.api.query_by_arch(arch, '200')
    




    