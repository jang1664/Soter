import numpy as np
import yaml
import os, sys
import copy
import math
from functools import reduce
from collections import defaultdict, OrderedDict
from subprocess import Popen, PIPE, call
import logging
import pathlib
import re
from concurrent.futures import ProcessPoolExecutor


class Timeloop(object):
    def __init__(self, in_config_path='./SpatialAccelerators', out_config_path='./out_config', accelerator='Simba',
                 opt_obj=None, use_sparse=False):

        self.accelerator = accelerator
        self.out_config_path = out_config_path
        self.use_sparse = use_sparse

        # export yaml file for timeloop
        with open(os.path.join(in_config_path, accelerator, 'arch.yaml'), 'r') as fd:
            self.arch = yaml.load(fd, Loader=yaml.SafeLoader)
        fd.close()
        with open(os.path.join(in_config_path, accelerator, 'problem.yaml'), 'r') as fd:
            self.problem = yaml.load(fd,Loader=yaml.SafeLoader)
        fd.close()
        with open(os.path.join(in_config_path, accelerator, 'mapspace.yaml'), 'r') as fd:
            self.mapspace = yaml.load(fd, Loader=yaml.SafeLoader)
        fd.close()

        self.opt_obj = opt_obj

        if self.use_sparse:
            with open(os.path.join(in_config_path, accelerator, 'sparse.yaml'), 'r') as fd:
                self.sparse = yaml.load(fd,Loader=yaml.SafeLoader)
            fd.close()

        # parse arch structure
        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels, num_pes = self.get_arch_info()
        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels
        self.num_pes = num_pes
        self._executable = 'timeloop-model'
        self.buf_energy_cost = self.get_default_buffer_energy_cost()
        self.density = {'Inputs': 0.5, 'Weights': 1, 'Outputs': 1}

        # self.dim_note = ['R', 'S', 'P', 'Q', 'C', 'K', 'H', 'N']
        # self.dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N', 7: 'H'}
        self.dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'H', 7: 'N'}
        print(self.dim2note.values())
        self.dimension, self.dimension_dict = self.get_problem_info()

        # get prime factors. Dictionary, key:dim_idx, value:dict[num : count]
        self.dimension_prime = {key: self.get_prime_factors(self.dimension_dict[key]) for key in self.dim2note.values()}

        # assign idx to each prime
        #TODO: why all dimension mixed?
        self.prime2idx = {}
        primes = set()
        for i, key in enumerate(self.dim2note.values()):
            tile_budget = self.dimension_prime[key]
            for k in tile_budget.keys():
                primes.add(int(k))
        primes = sorted(primes) # For all variables, gather prime factors.
        self.prime2idx = {'{}'.format(pf): i for i, pf in enumerate(primes)} # assign idx to each prime
        self.num_primes = len(self.prime2idx.keys())

        self.arch_path, self.problem_path, self.map_path, self.sparse_path, self.pool_path = [], [], [], [], []

    def get_default_buffer_energy_cost(self):
        buf_energy_cost = {'DRAM': 200,
                           'l2': 2.2,
                           'l1': 1.12,
                           'MAC': 1.0,
        }
        return buf_energy_cost

    def get_num_buffer_levels(self):
        return self.num_buffer_level

    def get_buffers_with_spmap(self):
        return self.buffers_with_spmap

    def get_buffer_spmap_cstr(self):
        return self.buffer_spmap_cstr

    def get_buffer_size_list(self):
        return self.buffer_size_list

    def get_prime_factors(self, n):
        primes = defaultdict(int)
        while n % 2 == 0:
            primes['2'] += 1
            n = n // 2
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                primes[f'{i}'] += 1
                n = n // i
        if n > 2:
            primes[f'{n}'] += 1
        return primes

    def get_factors(self, n):
        return list(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    def get_dimension_factors(self, dimension_dict):
        dimension_factors = dict()
        for key, value in dimension_dict.items():
            factors = self.get_factors(value)
            dimension_factors[key] = factors
        return dimension_factors

    def get_dimension_primes(self):
        return self.dimension, self.dimension_prime, self.prime2idx

    def get_problem_info(self):
        problem = copy.deepcopy(self.problem)
        dimension = []
        dimension_dicts = {}
        for key in self.dim2note.values():
            value = problem['problem']['instance'][key]
            dimension.append(value)
            dimension_dicts[key] = value
        return dimension, dimension_dicts

    def get_arch_info(self):
        arch = copy.deepcopy(self.arch)
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        arch = arch['architecture']

        if self.accelerator == 'Simba':
            main_memory = arch['subtree'][0]['local'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]['local'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe = arch['subtree'][0]['subtree'][0]['subtree'][0]
            num_pes = int(pe['name'].split('..')[1].split(']')[0]) + 1

            for buf in pe['local'][:-1]:
                buffer_name = buf['name']
                attributes = buf['attributes']
                depth = attributes['depth'] if 'depth' in attributes else float('Inf')
                word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
                width = attributes['width'] if 'width' in attributes else 8
                block_size = attributes['block-size'] if 'block-size' in attributes else 1
                buffer_size = depth * block_size
                instances = 1
                re_ret = re.search('.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                instances *= num_pes
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                num_buffer_levels += 1

            macc = pe['local'][-1]['name']
            instances = int(macc.split('..')[1].split(']')[0]) + 1
            instances *= num_pes
            num_instances.append(instances)
        elif 'Eyeriss' in self.accelerator:
            main_memory = arch['subtree'][0]['local'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]['local'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            dummy_buffer = arch['subtree'][0]['subtree'][0]['local'][1]
            buffer_name = dummy_buffer['name']
            attributes = dummy_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe = arch['subtree'][0]['subtree'][0]['subtree'][0]
            num_pes = int(pe['name'].split('..')[1].split(']')[0]) + 1

            for buf in pe['local'][:-1]:
                buffer_name = buf['name']
                attributes = buf['attributes']
                depth = attributes['depth'] if 'depth' in attributes else float('Inf')
                word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
                width = attributes['width'] if 'width' in attributes else 8
                block_size = attributes['block-size'] if 'block-size' in attributes else 1
                buffer_size = depth * block_size
                instances = 1
                re_ret = re.search('.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                instances *= num_pes
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                num_buffer_levels += 1

            macc = pe['local'][-1]['name']
            re_ret = re.search('.*\[', macc)
            if re_ret:
                instances = (int(macc.split('..')[1].split(']')[0]) + 1) * num_pes
            else:
                instances = num_pes
            num_instances.append(instances)
        elif 'TensorCore' in self.accelerator:
            main_memory = arch['subtree'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = main_memory['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = global_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            local_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]
            buffer_name = local_buffer['name']
            attributes = local_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = local_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]
            buffer_name = pe_buffer['name']
            attributes = pe_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = pe_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1
            num_pes = instances

            macc = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['name']
            re_ret = re.search('.*\[', macc)
            if re_ret:
                instances *= (int(macc.split('..')[1].split(']')[0]) + 1)
            num_instances.append(instances)

        print(buffer_name_list, num_instances, buffer_size_list)

        # calculate spatial loop constraint. If global buffer instance num is 2,
        # and local buffer instance num is 4, then spatial loop constraint is 2.
        sp_cstr = []
        for i in range(len(num_instances) - 1):
            allowed_sp_size = num_instances[i + 1] // num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i + 1] % num_instances[i] != 0:
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')

        print(sp_cstr)

        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
               num_buffer_levels, num_pes

    def get_dimension_dict(self, dim_value):
        return {note: value for note, value in zip(self.dim2note.values(), dim_value)}

    def get_tp_sp_tile_size(self, dim_value, sp_dim, sp_dim_value, timeloop_notation=True):
        """
          dim_value   : dict from dimension name to temporal tile size
          sp_dim      : spatial loop dimension names
          sp_dim_vale : dict from dimension name to spatial tile size
          time_notation : True if timeloop notation, False if numpy array
        """
        if timeloop_notation:
            temporal_series = []
            spatial_series = []
            for note, value in dim_value.items():
                if note not in sp_dim:
                    temporal_series.append(f'{note}={value}')
                    spatial_series.append(f'{note}=1')
                else:
                    # if current dimension is spatial loop dimension, tile size is divided by spatial tile size
                    sp_value = sp_dim_value[note]
                    tp_value = value // sp_value
                    temporal_series.append(f'{note}={tp_value}')
                    spatial_series.append(f'{note}={sp_value}')
            return ' '.join(temporal_series), ' '.join(spatial_series)
        else:
            temporal_series = []
            spatial_series = []
            for note in self.dim2note.values():
                if note not in sp_dim:
                    temporal_series.append(dim_value[note])
                    spatial_series.append(1)
                else:
                    sp_value = sp_dim_value[note]
                    tp_value = dim_value[note] // sp_value
                    temporal_series.append(tp_value)
                    spatial_series.append(sp_value)
            return np.array(temporal_series), np.array(spatial_series)

    def create_pool_env(self, num_pools):
        os.makedirs(self.out_config_path, exist_ok=True)
        arch_paths, problem_paths, map_paths, sparse_paths, pool_paths = [], [], [], [], []
        for i in range(num_pools):
            pool_dir = os.path.join(self.out_config_path, f'pool-{i}')
            os.makedirs(pool_dir, exist_ok=True)
            pool_paths.append(pool_dir)
            arch_paths.append(os.path.abspath(os.path.join(pool_dir, 'arch.yaml')))
            problem_paths.append(os.path.abspath(os.path.join(pool_dir, 'problem.yaml')))
            map_paths.append(os.path.abspath(os.path.join(pool_dir, 'map.yaml')))
            sparse_paths.append(os.path.abspath(os.path.join(pool_dir, 'sparse.yaml')))
        self.arch_path, self.problem_path, self.map_path, self.sparse_path, self.pool_path = arch_paths, problem_paths, map_paths, sparse_paths, pool_paths

    def get_problem_configs(self, dimension):
        problem = copy.deepcopy(self.problem)
        dimension_dict = self.get_dimension_dict(dimension)
        for key, value in dimension_dict.items():
            problem['problem']['instance'][key] = value
        if self.use_sparse:
            problem['problem']['instance']['densities'] = {}
            for key in ['Inputs', 'Weights', 'Outputs']:
                cur_density = self.density[key]
                if cur_density < 1:
                    problem['problem']['instance']['densities'][key] = {}
                    problem['problem']['instance']['densities'][key]['distribution'] = 'fixed-structured'
                    # problem['problem']['instance']['densities'][key]['distribution'] = 'hypergeometric'
                    problem['problem']['instance']['densities'][key]['density'] = cur_density
        return problem

    def get_map_config(self, program):
        """
        level : buffer level 
        step  : temporal or spatial step for a dimension
        """
        # make map.yaml from program tensor
        steps_per_level = len(self.dim2note.values())
        mapping = []
        # self.check_tile_fit_buffer(program)
        num_primes = len(self.prime2idx.keys())
        for level in range(1, self.num_buffer_level+1):
            target = self.buffer_name_list[f'l{level}']

            # 0 : order
            # 1~num_primes : temporal factor
            # num_primes+1~2*num_primes : spatial factor
            # it seems that num_primes+2 ~ is not meaningful
            level_program = program[(level-1)*steps_per_level:level*steps_per_level,:] # (steps_per_level, 2*num_primes+1)
            par_dims = set()
            perm_list = copy.deepcopy(list(self.dim2note.values())) # dimension name list
            tile_sizes_dict = {}
            sp_tile_sizes_dict = {}
            for i in range(steps_per_level):
                # note = dim2note[level_program[i, 0]]
                order = level_program[i, 0] # order of current step. current loop indvar order.
                note = self.dim2note[i] # note is dimension name like R, S, P, Q, C, K, H, N
                perm_list[order] = note # update order from dimension name like N- > M -> K ...
                if level_program[i, num_primes + 1] >= 1: # spatial loop is more than 1?
                    par_dims.add(note)

                # initialize tile size for current dimension. regardless of temporal or spatial
                tile_sizes_dict[note] = 1
                for k, v in self.prime2idx.items():
                    # temporal tile size calculation
                    # each step row have prime factor value count for each prime factor
                    # tilze size = 1 * (prime_factor_1^count_1) * (prime_factor_2^count_2) * ...
                    # if current prime factor is not related to current dimension, count will be 0. So tile size will be multiplied by 1.
                    tile_sizes_dict[note] *= pow(int(k), level_program[i, int(v) + 1])
                
                # spatial tile size calculation
                #TODO: why 2?. spatial tile size should be power of two??
                #TODO: why no loop? i think there are num_primes number spaitla factor counts
                sp_tile_sizes_dict[note] = pow(2, level_program[i, num_primes + 1])

            # temporal and spatial tile loop order must be same
            #TODO: why same? it seems hidden constraints
            permutation = ''
            for i in range(steps_per_level):
                permutation += perm_list[i]
            # print(perm_list)

            # make bypass map
            # bypass map is a map that bypass current buffer level. it comes from mapspace.yaml
            bypass_map = self.mapspace['mapspace']['constraints'][level - 1]

            # calculate temporal and spatial tile size
            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes_dict, par_dims, sp_tile_sizes_dict)

            # add mapping for current level
            # temporal, spatial, bypass
            cur_map = {'target': target,
                       'type': 'temporal',
                        'factors': tp_tile_sizes,
                       'permutation': permutation,
                       }
            mapping.append(cur_map)
            if f'l{level}' in self.buffers_with_spmap:
                cur_map = {'target': target,
                           'type': 'spatial',
                           'factors': sp_tile_sizes,
                           'permutation': permutation,
                           }
                mapping.append(cur_map)
            mapping.append(bypass_map)
        return {'mapping': mapping}

    def get_configs(self, dimension, program):
        arch = self.arch
        problem = self.get_problem_configs(dimension)
        map = self.get_map_config(program)
        return arch, problem, map

    def write_config(self, arch, problem, map, arch_path, problem_path, map_path, sparse_path=None):
        with open(arch_path, 'w') as fd:
            yaml.dump(arch, fd)
        fd.close()
        with open(problem_path, 'w') as fd:
            yaml.dump(problem, fd)
        fd.close()
        with open(map_path, 'w') as fd:
            yaml.dump(map, fd)
        fd.close()
        if self.use_sparse:
            with open(sparse_path, 'w') as fd:
                yaml.dump(self.sparse, fd)
            fd.close()

    def thread_fun(self, args):
        # args : program for current sample
        program, pool_idx = args
        arch, problem, map = self.get_configs(self.dimension, program)
        self.write_config(arch, problem, map, arch_path=self.arch_path[pool_idx],
                          problem_path=self.problem_path[pool_idx], map_path=self.map_path[pool_idx], sparse_path=self.sparse_path[pool_idx])
        command = [self._executable, self.arch_path[pool_idx], self.problem_path[pool_idx], self.map_path[pool_idx]]
        if self.use_sparse:
            command += [self.sparse_path[pool_idx]]
        process = Popen(command, stdout=PIPE, stderr=PIPE, cwd=self.pool_path[pool_idx])
        stdout, stderr = process.communicate()
        process.wait()
        if stderr:
            print("stderrstderr: ", stderr, program)
            return [-float('Inf')] * len(self.opt_obj)
        else:
            try:
                stats = self.run_config(self.pool_path[pool_idx])
                fitness = self.judge(stats, self.opt_obj)
            except Exception as e:
                print("Exception: ", e)
                fitness = [-float('Inf')] * len(self.opt_obj)
            return fitness

    def run(self, programs):
        # program : final_program_seq
        num_samples = programs.shape[0]
        pool = ProcessPoolExecutor(num_samples)
        # pool = None
        self.create_pool_env(num_pools=num_samples)

        fitness = np.ones((num_samples, len(self.opt_obj))) * np.NINF

        if not pool:
            for i, program in enumerate(programs):
                fit = self.thread_fun((program, 0))
                fitness[i] = fit
        else:
            while (1):
                try:
                    fits = list(pool.map(self.thread_fun, zip(programs, np.arange(len(programs)))))
                    for i, fit in enumerate(fits):
                        fitness[i] = fit
                    break
                except Exception as e:
                    print(type(e).__name__, e)
                    pool.shutdown(wait=False)
                    pool = ProcessPoolExecutor(num_samples)

        return fitness

    def judge(self, stats, opt_obj='all'):
        if opt_obj == 'all':
            opt_obj = ['edp', 'latency', 'energy']
        ret = []

        for f in opt_obj:
            if f == 'edp':
                ret.append(-stats['cycles'] * stats['energy']) # energy_uJ
            if f == 'latency':
                ret.append(-stats['cycles'])
            # if f == 'utilization':
            #     ret.append(stats['utilization'])
            if f == 'energy':
                ret.append(-stats['energy'] ) # energy_uJ
        return ret

    def run_config(self, filename):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # capture everything

        # Output file names.
        out_prefix = "timeloop-model."
        report_prefix = out_prefix + 'stats.txt'
        xml_file_name = out_prefix + "map+stats.xml"

        filename = pathlib.Path(filename).resolve()
        report_file = filename / report_prefix
        status_dict = dict()
        if report_file.exists():
            with open(report_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                m = re.match(r"Energy: (.*) uJ", line)
                if m:
                    energy = m.group(1)
                    status_dict['energy'] = float(energy)
                else:
                    # m = re.match(r"Max topology cycles: (.*)", line)
                    m = re.match(r"Cycles: (.*)", line)
                    if m:
                        cycle = m.group(1)
                        status_dict['cycles'] = int(cycle)
        return status_dict

