##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###################################################################
# BOHB: Robust and Efficient Hyperparameter Optimization at Scale #
# required to install hpbandster ##################################
# pip install hpbandster         ##################################
###################################################################
# bash ./scripts-search/algos/BOHB.sh -1         ##################
###################################################################
import os, sys, time, random, argparse
from copy import deepcopy
from pathlib import Path
#import torch
lib_dir = (Path(__file__).parent / '..' / '..' / 'nas201bench').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

#from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger
from log_utils    import AverageMeter, time_string, convert_secs2time
#from nas_201_api  import NASBench201API as API
from lookup       import NAS201Benchmark

# BOHB: Robust and Efficient Hyperparameter Optimization at Scale, ICML 2018
import ConfigSpace
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker


def get_configuration_space(max_nodes, search_space):
  cs = ConfigSpace.ConfigurationSpace()
  #edge2index   = {}
  for i in range(1, max_nodes):
    for j in range(i):
      node_str = '{:}<-{:}'.format(i, j)
      cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
  return cs



class MyWorker(Worker):

  def __init__(self, *args, nas_bench=None, time_budget=None, **kwargs):
    super().__init__(*args, **kwargs)
    
    self._dataname      = nas_bench.dataset
    self._nas_bench     = nas_bench
    self.time_budget    = time_budget
    self.seen_archs     = []
    self.sim_cost_time  = 0
    self.real_cost_time = 0
    self.is_end         = False

  def get_the_best(self):
    assert len(self.seen_archs) > 0
    best_index, best_acc = -1, None
    for arch_index in self.seen_archs:
      info = self._nas_bench.get_eval_result(arch_index)
      vacc = info['valid-accuracy']
      if best_acc is None or best_acc < vacc:
        best_acc = vacc
        best_index = arch_index
    assert best_index != -1
    return best_index

  def compute(self, config, budget, **kwargs):
    start_time = time.time()
    
    arch_index = self._nas_bench.get_arch_index( config )
    if arch_index > 0:
      info       = self._nas_bench.get_eval_result(arch_index)
      cur_time   = info['train-all-time'] + info['valid-per-time']
      cur_vacc   = info['valid-accuracy']
      self.real_cost_time += (time.time() - start_time)
      if self.sim_cost_time + cur_time <= self.time_budget and not self.is_end:
        self.sim_cost_time += cur_time
        self.seen_archs.append( arch_index )
        return ({'loss': 1.0 - float(cur_vacc / 100),
                'info': {'seen-arch'     : len(self.seen_archs),
                          'sim-test-time' : self.sim_cost_time,
                          'current-arch'  : arch_index}
              })
      else:
        self.is_end = True
        return ({'loss': 1.0,
                'info': {'seen-arch'     : len(self.seen_archs),
                          'sim-test-time' : self.sim_cost_time,
                          'current-arch'  : None}
              })
    else:
      self.is_end = True
      return ({'loss': 1.0,
              'info': {'seen-arch'     : len(self.seen_archs),
                        'sim-test-time' : self.sim_cost_time,
                        'current-arch'  : None}
            })          


def main(xargs):
  #assert torch.cuda.is_available(), 'CUDA is not available.'
  #torch.backends.cudnn.enabled   = True
  #torch.backends.cudnn.benchmark = False
  #torch.backends.cudnn.deterministic = True
  #torch.set_num_threads( xargs.workers )
  
  #config_path = 'nas201bench/R-EA.config'
  #config = load_config(config_path, None, logger)
  #logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))
  #extra_info = {'config': config, 'train_loader': None, 'valid_loader': None}
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(xargs)

  # Preparing NAS201bench
  nas_bench = NAS201Benchmark(xargs.dataset)
  max_nodes = nas_bench.max_nodes
  search_space = nas_bench.get_search_space()
  cs = get_configuration_space(max_nodes, search_space)
  

  hb_run_id = '0'
  NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
  ns_host, ns_port = NS.start()
  num_workers = 1

  workers = []
  for i in range(num_workers):
    w = MyWorker(nameserver=ns_host, 
                nameserver_port=ns_port, 
                nas_bench=nas_bench, 
                time_budget=xargs.time_budget, 
                run_id=hb_run_id, 
                id=i)
    w.run(background=True)
    workers.append(w)

  start_time = time.time()
  bohb = BOHB(configspace=cs,
            run_id=hb_run_id,
            eta=3, min_budget=12, 
            max_budget=200,
            nameserver=ns_host,
            nameserver_port=ns_port,
            num_samples=xargs.num_samples,
            random_fraction=xargs.random_fraction, 
            bandwidth_factor=xargs.bandwidth_factor,
            ping_interval=10, 
            min_bandwidth=xargs.min_bandwidth)
  
  results = bohb.run(xargs.n_iters, min_n_workers=num_workers)

  bohb.shutdown(shutdown_workers=True)
  NS.shutdown()

  real_cost_time = time.time() - start_time

  id2config = results.get_id2config_mapping()
  incumbent = results.get_incumbent_id()
  logger.log('Best found configuration: {:} within {:.3f} s'.format(id2config[incumbent]['config'], real_cost_time))

  best_arch = nas_bench.convert_structure( id2config[incumbent]['config'] )

  info = nas_bench.query_by_arch(best_arch)
  if info is None: logger.log('Did not find this architecture : {:}.'.format(best_arch))
  else           : logger.log('{:}'.format(info))
  logger.log('-'*100)

  logger.log('workers : {:.1f}s with {:} archs'.format(workers[0].time_budget, 
                                                len(workers[0].seen_archs)))
  logger.close()
  #return logger.log_dir, nas_bench.query_index_by_arch( best_arch ), real_cost_time


if __name__ == '__main__':
  parser = argparse.ArgumentParser("BOHB: Robust and Efficient Hyperparameter Optimization at Scale")
  parser.add_argument('--dataset',            default="ImageNet16-120", type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--time_budget',        type=int,   help='The total time cost budge for searching (in seconds).')
  # BOHB
  parser.add_argument('--min_bandwidth',    default=.3,  type=float, nargs='?', help='minimum bandwidth for KDE')
  parser.add_argument('--num_samples',      default=4,  type=int, nargs='?', help='number of samples for the acquisition function')
  parser.add_argument('--random_fraction',  default=.0, type=float, nargs='?', help='fraction of random configurations')
  parser.add_argument('--bandwidth_factor', default=3,   type=int, nargs='?', help='factor multiplied to the bandwidth')
  parser.add_argument('--n_iters',          default=100, type=int, nargs='?', help='number of iterations for optimization method')
  # log
  parser.add_argument('--workers',            type=int,   default=1,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--print_freq',         default=200, type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          default=1, type=int, help='manual seed')
  args = parser.parse_args()
  
  main(args)
