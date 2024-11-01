import os
import math
import random
import pickle
import shutil
import glob
from datetime import datetime
import pandas as pd

from torch.distributions import Categorical
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from cost_model import Timeloop
from analytical_model import Model

import argparse

parser = argparse.ArgumentParser(description='Program Tuner')
parser.add_argument('--random_factor', action='store_true', help='Random factor sample')
parser.add_argument('--random_order', action='store_true', help='Random order sample')
args, unknown = parser.parse_known_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ProgramTransformer(nn.Module):
    def __init__(self, operator_instance, steps_per_level, tile_size, prime2idx, buffer_size_list, buf_spmap_cstr):
        super().__init__()

        self.operator_instance = operator_instance
        self.steps_per_level = steps_per_level
        self.prime2idx = prime2idx
        self.idx2prime = {value: key for key, value in prime2idx.items()}
        self.num_primes = len(self.prime2idx.keys())

        self.buffer_size_list = buffer_size_list
        self.buf_spmap_cstr = buf_spmap_cstr

        self.transformer = Transformer(d_word_vec=512, d_model=512, d_inner=1024,
                                       n_layers=4, n_head=8, d_k=64, d_v=64, dropout=0,
                                       n_position=100, trg_emb_prj_weight_sharing=True,
                                       scale_emb_or_prj='prj',
                                       order_size=steps_per_level, tile_size=tile_size,
                                       num_primes=len(self.prime2idx.keys()))

    def forward(self, program_seq, order_mask, tile_remain_budgets, tile_masks, cur_buffer_level,
                loop_ind, remain_buffer_size, tile2_max, max_temporal_tile2, sp_tile2_max, sp_tile2_min):
        """
        This function is called for each step. If buffer level 3, step per level 2, this function will be called (3*2) times 

        Parameters:
          program_seq
            program sequence that currently decided.
          order_mask:
            mask for order. If order 3 was selected, then order_mask will make prob of 3 to zero
          tile_remain_budgets
          tile_masks
          cur_buffer_level

        Returns:
          order_action:
            order of program.
          tile_actions:
            temporal tiling factor of each prime factor
          sp_tile_actions:
          log_probs:
            order prob, temporal tiling factor prob, spatial tiling factor prob
          log_prob_masks:
        """

        order_logit, tile_logits, sp_tile2_logit = self.transformer(program_seq)
        tile2_logit = tile_logits[:, 0]
        num_samples = program_seq.size(0)

        # order_action = program_seq.new_ones(num_samples).fill_(loop_ind)
        # order_log_prob = tile2_logit.new_zeros(num_samples)
        # order_log_prob_mask = tile2_logit.new_zeros(num_samples)

        # sample order of current (loop indvar, level)
        if args.random_order:
          order_score = torch.ones_like(order_logit) + order_mask
        else:
          order_score = order_logit + order_mask # order mask : make -inf for invalid options
        order_prob = F.softmax(order_score, dim=-1)
        order_density = Categorical(order_prob)
        order_action = order_density.sample()
        order_log_prob = order_density.log_prob(order_action)
        order_log_prob_mask = ((order_mask == 0).sum(dim=-1) > 1).float()

        log_probs = tile2_logit.new_zeros(num_samples, 1 + self.num_primes + 1)
        log_prob_masks = tile2_logit.new_zeros(num_samples, 1 + self.num_primes + 1)

        log_probs[:, 0] = order_log_prob
        log_prob_masks[:, 0] = order_log_prob_mask

        if cur_buffer_level == len(self.buffer_size_list):
            return order_action, None, None, log_probs, log_prob_masks

        tile2_mask = tile_masks[:, loop_ind, 0]

        sp_tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            sp_tile2_mask_tmp[np.arange(num_samples), sp_tile2_max + i] = float('-inf')
        sp_tile2_mask_tmp = sp_tile2_mask_tmp[:, :tile2_mask.size(-1)]

        sp_tile2_mask_tmp = torch.cat([torch.zeros_like(tile2_mask), sp_tile2_mask_tmp], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            sp_tile2_mask_tmp[np.arange(num_samples), sp_tile2_min + tile2_mask.size(-1) - i] = float('-inf')
        sp_tile2_mask = sp_tile2_mask_tmp[:, tile2_mask.size(-1):]

        # sample spatial tiling factor of current (loop indvar, level)
        if not args.random_factor:
          sp_tile2_score = sp_tile2_logit + sp_tile2_mask
        else:
          sp_tile2_score = torch.ones_like(sp_tile2_logit) + sp_tile2_mask
        sp_tile2_prob = F.softmax(sp_tile2_score, dim=-1)
        sp_tile2_density = Categorical(sp_tile2_prob)
        sp_tile2_action = sp_tile2_density.sample()
        sp_tile2_log_prob = sp_tile2_density.log_prob(sp_tile2_action)
        sp_tile2_log_prob_mask = ((sp_tile2_mask == 0).sum(dim=-1) > 1).float()

        # if cur_buffer_level == 4:
        #     print(mode, sp_tile2_action[0], sp_tile2_max[0], tile2_max[0])

        tile2_min = sp_tile2_action
        tile2_max = torch.minimum(tile2_max, tile2_min + max_temporal_tile2.long())

        tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            tile2_mask_tmp[np.arange(num_samples), tile2_max + i] = float('-inf')
        tile2_mask_tmp = tile2_mask_tmp[:, :tile2_mask.size(-1)]

        tile2_mask_tmp = torch.cat([torch.zeros_like(tile2_mask), tile2_mask_tmp], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            tile2_mask_tmp[np.arange(num_samples), tile2_min + tile2_mask.size(-1) - i] = float('-inf')
        tile2_mask = tile2_mask_tmp[:, tile2_mask.size(-1):]

        # sample temporal tiling factor of current (loop indvar, level)
        if not args.random_factor:
          tile2_score = tile2_logit + tile2_mask
        else:
          tile2_score = torch.ones_like(tile2_logit) + tile2_mask
        tile2_prob = F.softmax(tile2_score, dim=-1)
        tile2_density = Categorical(tile2_prob)
        tile2_action = tile2_density.sample()
        tile2_log_prob = tile2_density.log_prob(tile2_action)
        tile2_log_prob_mask = ((tile2_mask == 0).sum(dim=-1) > 1).float()

        tile_action = tile2_action
        tile_actions = []
        sp_tile_actions = []
        log_probs = []
        log_prob_masks = []

        log_probs.append(order_log_prob)
        log_prob_masks.append(order_log_prob_mask)

        tile_actions.append(tile2_action)
        sp_tile_actions.append(sp_tile2_action)
        log_probs.append(tile2_log_prob)
        log_prob_masks.append(tile2_log_prob_mask)
        for p in range(1, self.num_primes): #TODO: prime??
            remain_buffer_size = remain_buffer_size / torch.pow(int(self.idx2prime[p - 1]), tile_action).float()
            tile_max = torch.log2(torch.clamp(remain_buffer_size, min=1)) / math.log2(int(self.idx2prime[p]))
            tile_max = torch.clamp(tile_max.long(), min=0, max=tile_masks.size(-1) - 1)
            tile_max = torch.minimum(tile_max, tile_remain_budgets[:, loop_ind, p])

            tile_mask = copy.deepcopy(tile_masks[:, loop_ind, p])
            tile_mask_tmp = torch.cat([tile_mask, torch.zeros_like(tile_mask)], dim=-1)
            for i in range(1, tile_mask.size(-1) + 1):
                tile_mask_tmp[np.arange(num_samples), tile_max + i] = float('-inf')
            tile_mask = tile_mask_tmp[:, :tile_mask.size(-1)]

            tile_logit = tile_logits[:, p]
            if not args.random_factor:
              tile_score = tile_logit + tile_mask
            else:
              tile_score = torch.ones_like(tile_logit) + tile_mask
            tile_prob = F.softmax(tile_score, dim=-1)
            tile_density = Categorical(tile_prob)
            tile_action = tile_density.sample()
            tile_log_prob = tile_density.log_prob(tile_action)
            tile_log_prob_mask = ((tile_mask == 0).sum(dim=-1) > 1).float()

            tile_actions.append(tile_action)
            log_probs.append(tile_log_prob)
            log_prob_masks.append(tile_log_prob_mask)
            sp_tile_actions.append(program_seq.new_zeros(num_samples))

        log_probs.append(sp_tile2_log_prob)
        log_prob_masks.append(sp_tile2_log_prob_mask)

        tile_actions = torch.stack(tile_actions, dim=1)
        sp_tile_actions = torch.stack(sp_tile_actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        log_prob_masks = torch.stack(log_prob_masks, dim=1)

        return order_action, tile_actions, sp_tile_actions, log_probs, log_prob_masks


class Tuner:

    def __init__(self, operator_instance, accelerator, report_dir, optim_obj):
        super().__init__()

        self.opt_obj = [optim_obj, 'latency', 'energy']

        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.report_dir = report_dir

        self.accelerator = accelerator
        self.cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path=self.timeloop_out_config_path,
                                   accelerator=accelerator, opt_obj=self.opt_obj)

        # dict from dimension string name to int idx
        self.dim2note = self.cost_model.dim2note
        self.len_dimension = len(self.dim2note.values())

        self.num_buf_levels = self.cost_model.get_num_buffer_levels()
        # print(f'Number of buffer levels: {self.num_buf_levels}')
        self.buffer_size_list = self.cost_model.get_buffer_size_list()
        self.buf_spmap_cstr = self.cost_model.get_buffer_spmap_cstr()
        self.buffers_with_spmap = list(self.cost_model.get_buffers_with_spmap())
        self.operator_instance = operator_instance
        self.dimension, self.dimension_prime, self.prime2idx = self.cost_model.get_dimension_primes()
        self.num_primes = len(self.prime2idx.keys())
        # print(self.buf_spmap_cstr, self.buffers_with_spmap, self.buffer_size_list, self.prime2idx)
        self.idx2prime = {value: key for key, value in self.prime2idx.items()}

        self.best_obj_record = []
        self.best_latency_record = []
        self.best_energy_record = []
        self.best_program_record = []
        self.best_obj = float("-Inf")
        self.best_latency = float("-Inf")
        self.best_energy = float("-Inf")
        self.best_program = None
        self.worst_obj = None

        # level -> buffer level
        #TODO: why two one?? it seems barrier
        # it means we need specific level order for each arch. bad..!
        if 'Simba' in self.accelerator:
            self.level_order = [1, 2, 3, 4, 5, 6, 1]
        elif 'Eyeriss' in self.accelerator:
            self.level_order = [4, 5, 1, 2, 3, 6, 1]
        elif 'TensorCore' in self.accelerator:
            self.level_order = [2, 3, 1, 4, 1]

        # each step means dimension
        self.steps_per_level = len(self.dim2note.values())
        self.total_steps = self.num_buf_levels * self.steps_per_level

        #TODO: is it good? constant?
        self.num_samples = 32
        self.max_tile = 30

        # order mask will make prob of already selected order to zero
        # For each level, there are multiple steps(dimensions). we sholuld select order of each dimensions
        # If 5 dimensions, order will be 0~4. if order 3 was selected, then order mask will make prob of 3 to zero
        # soter assume temporal and spatial dimension order sholud be same, So there is only one order mask
        # element idx means order prob. if prob vector is (1 x N) vector, [0,0] is prob of order 0
        # transformer decoder generate steps_per_level + 1 length vector, So we add +1 for order_mask
        self.initial_order_mask = np.zeros((self.num_samples, self.len_dimension + 1), dtype=float)
        self.initial_order_mask[:, -1] = float('-inf')

        # tile budget means power number of each prime factor. If X = 2^5 * 3^3, then tile budget of 2 is 5, 3 is 3
        # tile mask will make prob of not valid tile size. not valid one is decided by series of constraints like remaining ...buffer size, spatial resource size
        # 0:self.max_tile+1 will be used by explorer. But we allocate more space..?
        self.tile_budgets = np.zeros((self.num_samples, self.len_dimension, self.num_primes), dtype=np.int32)
        self.initial_tile_masks = np.zeros(
            (self.num_samples, self.len_dimension, self.num_primes, (self.max_tile + 1) * 2), dtype=float)

        # print("buffers_with_spmap", self.buffers_with_spmap)
        for i, key in enumerate(self.dim2note.values()):
            tile_budget = self.dimension_prime[key]
            for k, v in self.prime2idx.items():
                self.tile_budgets[:, i, v] = tile_budget[k]

                # For each factor, we should mask out bigger than max tile size power factor
                self.initial_tile_masks[:, i, v, tile_budget[k] + 1:] = float('-inf')

        # program seq is decisions for each step. For each step, we should decide order, tile size, spatial tile size
        # 1 -> order, num_primes -> tile size, num_primes -> spatial tile size
        # we make start token. order is steps_per_level, tile size is max_tile, spatial tile size is max_tile for every prime factor
        # for different steps, order same for step idx, tile size power factor is tile budget for last level, spatial tile size is 0
        # +1 at total step + 1 is for start token(?)
        length = self.num_primes * 2 + 1
        self.initial_program_seq = np.zeros((self.num_samples, self.total_steps + 1, length), dtype=np.int32)
        self.initial_program_seq[:, 0, 0] = self.steps_per_level
        self.initial_program_seq[:, 0, 1: self.num_primes + 1] = self.max_tile
        self.initial_program_seq[:, 0, self.num_primes + 1: self.num_primes + 1 + self.num_primes] = self.max_tile

        # For first, spatial expand factor is one and temporal tile budget is assigned to dram only
        for i in range(self.num_buf_levels):
            for j in range(self.steps_per_level):
                self.initial_program_seq[:, i * self.steps_per_level + j + 1, 0] = j
                self.initial_program_seq[:, i * self.steps_per_level + j + 1, self.num_primes + 1:] = 0
                if i == self.num_buf_levels - 1:
                    self.initial_program_seq[:, i * self.steps_per_level + j + 1, 1: self.num_primes + 1] = self.tile_budgets[:, j]
                else:
                    self.initial_program_seq[:, i * self.steps_per_level + j + 1, 1: self.num_primes + 1] = 0

        self.finished_levels = []

        set_seed(42)
        self.analytical_model = Model(self.prime2idx, self.buffer_size_list, self.buf_spmap_cstr, self.steps_per_level,
                                      self.operator_instance, self.accelerator)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.explorer = ProgramTransformer(self.operator_instance, self.steps_per_level, self.max_tile, self.prime2idx,
                                           self.buffer_size_list, self.buf_spmap_cstr).to(device)
        self.optimizer = ScheduledOptim(
            optim.Adam(self.explorer.parameters(), betas=(0.9, 0.98), eps=1e-09),
            lr_mul=0.01, d_model=512, n_warmup_steps=2)

    def run(self, epochs):
        record = {}
        arch = self.cost_model.arch
        problem = self.cost_model.get_problem_configs(self.cost_model.dimension)
        record["arch"] = arch
        record["problem"] = problem
        record["map_record"] = {
            "epoch" : [],
            "map" : [],
            "batch_idx" : [],
            "cycle" : [],
            "energy" : [],
            "edp" : []
        }
        record["loss"] = []

        self.explorer.train()
        for ep in range(epochs):
            print('Epoch {}'.format(ep))

            final_program_seq, total_rewards, total_log_probs, total_log_prob_masks = self.exploration()

            fitness = self.cost_model.run(final_program_seq[:, :, :])
            latency = fitness[:, 1]
            energy = fitness[:, 2]
            obj_values = fitness[:, 0]

            # log records
            for i in range(self.num_samples):
              map = self.cost_model.get_map_config(final_program_seq[i])
              record["map_record"]["epoch"].append(ep)
              record["map_record"]["map"].append(map)
              record["map_record"]["batch_idx"].append(i)
              record["map_record"]["cycle"].append(latency[i])
              record["map_record"]["energy"].append(energy[i])
              record["map_record"]["edp"].append(latency[i] * energy[i])

            if args.random_factor and args.random_order:
              loss = None
            else:
              loss = self.optimization(obj_values, total_rewards, total_log_probs, total_log_prob_masks)
              record["loss"].append(loss.item())
            chkpt = self.record_chkpt(ep == epochs - 1)
            best_idx = np.argmax(obj_values)
            if obj_values[best_idx] > self.best_obj:
                self.best_obj = obj_values[best_idx]
                self.best_latency = latency[best_idx]
                self.best_energy = energy[best_idx]
                self.best_program = final_program_seq[best_idx]
                self.create_timeloop_report(self.best_program, self.report_dir)
            print("Achieved obj: ", self.best_obj, obj_values[best_idx], self.best_latency,
                  self.best_energy, (obj_values > float('-inf')).sum())
        self.clean_timeloop_output_files()

        pickle.dump(record, open(os.path.join(self.report_dir, 'record.pkl'), 'wb'))
        return chkpt

    def exploration(self):   # Transformer-based
        """
      
        Returns
        -------
        final_program_seq.cpu().numpy()
          tile size, order for total steps
        total_rewards
        total_log_probs
        total_log_prob_masks
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.finished_levels = []
        total_log_probs = []
        total_log_prob_masks = []
        total_rewards = []

        order_mask = torch.from_numpy(copy.deepcopy(self.initial_order_mask)).type(torch.FloatTensor).to(device)
        tile_remain_budgets = torch.from_numpy(copy.deepcopy(self.tile_budgets)).type(torch.LongTensor).to(device)
        tile_masks = torch.from_numpy(copy.deepcopy(self.initial_tile_masks)).type(torch.FloatTensor).to(device)

        program_seq = torch.from_numpy(copy.deepcopy(self.initial_program_seq[:, :1, :])).type(torch.LongTensor).to(device) # start from start token
        program_seq_disorder = torch.from_numpy(copy.deepcopy(self.initial_program_seq[:, 1:, :])).type(torch.LongTensor).to(device) # start from first step to end 
        final_program_seq = torch.from_numpy(copy.deepcopy(self.initial_program_seq[:, 1:, :])).type(torch.LongTensor).to(device)

        # explore buffer level. order is set by user
        start_level_order = 0
        cur_buffer_level = self.level_order[start_level_order]

        # mode is step number including buffer level
        # mode will be incremented by 1 after each step. So order, tile size, spatial tile size will be decided step by step
        # exploration order of buffer level will be decided by level order (design time parameter)
        # when exploration of current step, program sequence will be updated.
        mode = (cur_buffer_level - 1) * self.steps_per_level

        for time_step in range(self.total_steps):
            # loop_ind is dimension id for each buffer level
            loop_ind = mode % self.steps_per_level

            # calculate current step constraints
            remain_buffer_size, tile2_max, max_temporal_tile2, sp_tile2_max, sp_tile2_min = self.analysis(
                program_seq_disorder, tile_remain_budgets, mode, cur_buffer_level, loop_ind)

            # get order, tile size, spatial tile size for current step
            step_order, step_tile, step_parallel, step_log_prob, step_log_prob_mask = self.explorer(
                program_seq, order_mask, tile_remain_budgets, tile_masks[:, :, :, 0:self.max_tile + 1], cur_buffer_level,
                loop_ind, remain_buffer_size, tile2_max, max_temporal_tile2, sp_tile2_max, sp_tile2_min)

            if cur_buffer_level < self.num_buf_levels:
                length = self.num_primes * 2 + 1 # order + temporal tile + spatial tile
                cur_seq = torch.zeros((self.num_samples, 1, length), dtype=torch.long, device=device)
                cur_seq[:, 0, 0] = step_order
                cur_seq[:, 0, 1: self.num_primes + 1] = step_tile
                cur_seq[:, 0, self.num_primes + 1:] = step_parallel

                # concat current step decision to previous seq
                program_seq = torch.cat((program_seq, cur_seq), dim=1)

                # make program seq without order exploration.
                #TODO: seq_disorder_ind is same as mode?
                seq_disorder_ind = (cur_buffer_level - 1) * self.steps_per_level + loop_ind
                program_seq_disorder[:, seq_disorder_ind, 1: self.num_primes + 1] = step_tile
                program_seq_disorder[:, seq_disorder_ind, self.num_primes + 1:] = step_parallel

                # set final program seq
                final_program_seq[:, mode, 0] = step_order
                final_program_seq[:, mode, 1: self.num_primes + 1] = step_tile
                final_program_seq[:, mode, self.num_primes + 1:] = step_parallel

                # update order mask
                order_mask[torch.arange(self.num_samples), step_order] = float('-inf')

                # update tile budget and tile mask considering current budget
                tile_remain_budgets[:, loop_ind] -= step_tile

                for i in range(1, self.max_tile + 1):
                    for j in range(0, self.num_primes):
                        tile_remain_budget = tile_remain_budgets[:, loop_ind, j]
                        tile_masks[np.arange(self.num_samples), loop_ind, j, tile_remain_budget + i] = float('-inf')
            else:
                # if current level is last level, just use remain tile budget. spatial is always zero..?
                step_tile = tile_remain_budgets[:, loop_ind]
                step_parallel = torch.zeros((self.num_samples, 1), dtype=torch.long, device=device)

                length = self.num_primes * 2 + 1
                cur_seq = torch.zeros((self.num_samples, 1, length), dtype=torch.long, device=device)
                cur_seq[:, 0, 0] = step_order
                cur_seq[:, 0, 1: self.num_primes + 1] = step_tile
                cur_seq[:, 0, self.num_primes + 1:] = step_parallel

                program_seq = torch.cat((program_seq, cur_seq), dim=1)

                seq_disorder_ind = (cur_buffer_level - 1) * self.steps_per_level + loop_ind
                program_seq_disorder[:, seq_disorder_ind, 1 : self.num_primes + 1] = step_tile
                program_seq_disorder[:, seq_disorder_ind, self.num_primes + 1:] = step_parallel

                final_program_seq[:, mode, 0] = step_order
                final_program_seq[:, mode, 1: self.num_primes + 1] = step_tile
                final_program_seq[:, mode, self.num_primes + 1:] = step_parallel

                order_mask[torch.arange(self.num_samples), step_order] = float('-inf')

            # update mode. if mode is multiple of steps_per_level, then move to next buffer level and 
            # reset order_mask
            # append finished level
            mode += 1
            if mode % self.steps_per_level == 0:
                start_level_order += 1
                cur_buffer_level = self.level_order[start_level_order]
                mode = (cur_buffer_level - 1) * self.steps_per_level
                order_mask = torch.from_numpy(copy.deepcopy(self.initial_order_mask)).type(torch.FloatTensor).to(device)
                self.finished_levels.append(cur_buffer_level)

            # rewared is sparse.
            total_rewards.append(np.zeros(self.num_samples))
            total_log_probs.append(step_log_prob)
            total_log_prob_masks.append(step_log_prob_mask)

        return final_program_seq.cpu().numpy(), total_rewards, total_log_probs, total_log_prob_masks

    def analysis(self, program_seq_disorder, tile_remain_budgets, mode, cur_buffer_level, loop_ind):
        """
        Parameters
        ----------
        program_seq_disorder:
          program sequence. each step has own temporal factor, spatial factor, order in current buffer level 
          all tile and spatial factor of previous mode is decided by explorer.

        tile_remain_budgets:
          For each dimension, how many prime factor remains. It include all level
        mode:

        cur_buffer_level:

        loop_ind:

        Returns
        -------
        remain_buffer_size:
        tile2_max:
        max_temporal_tile2:
        sp_tile2_max:
        sp_tile2_min:
        """

        # get remain buffer size for current loop induce variable and current buffer level
        # we sholud consider current and later buffer level constraints simultaneously
        remain_buffer_size = self.analytical_model.get_remain_buffer_size(cur_buffer_level, program_seq_disorder,
                                                                          loop_ind)
        for later_level in range(cur_buffer_level + 1, len(self.buffer_size_list) + 1):
            remain_buffer_size = torch.minimum(remain_buffer_size,
                                               self.analytical_model.get_remain_buffer_size(later_level,
                                                                                            program_seq_disorder,
                                                                                            loop_ind))
        # get max tile size of power of 2 factor for current loop induce variable and current buffer level
        tile2_max = torch.log2(torch.clamp(remain_buffer_size, min=1)) # how can many multiple from previous tile size 
        tile2_max = torch.clamp(tile2_max.long(), min=0)
        tile2_remain_budgets = tile_remain_budgets[:, :, 0]
        tile2_max = torch.minimum(tile2_max, tile2_remain_budgets[:, loop_ind])

        remain_buf_spmap = self.analytical_model.get_remain_buf_spmap(cur_buffer_level, program_seq_disorder)
        tile2_remain_dimension_budgets = tile2_remain_budgets.sum(dim=-1) # sum of power of 2 factor count for every dimension
        max_temporal_tile2 = self.analytical_model.get_max_temporal_size(cur_buffer_level, self.finished_levels,
                                                                         tile2_remain_dimension_budgets,
                                                                         remain_buf_spmap)

        sp_tile2_max, sp_tile2_min = self.analytical_model.get_spatial_size(mode, tile2_max, loop_ind,
                                                                            tile2_remain_budgets, remain_buf_spmap)

        return remain_buffer_size, tile2_max, max_temporal_tile2, sp_tile2_max, sp_tile2_min

    def optimization(self, obj_values, total_rewards, total_log_probs, total_log_prob_masks):  # policy gradient

        self.optimizer.zero_grad()

        obj_values[obj_values == float('-inf')] = float('inf')
        if self.worst_obj is None:
            self.worst_obj = obj_values.min()
        else:
            self.worst_obj = min(self.worst_obj, obj_values.min())
        obj_values[obj_values == float('inf')] = self.worst_obj * 2

        sort_idx = np.argsort(obj_values)
        top_k_idx = sort_idx[int(self.num_samples / 4) - 1]
        rewards = (obj_values - obj_values[top_k_idx])

        # rewards = (obj_values - self.worst_obj)

        total_rewards[-1] = rewards

        total_rewards = np.stack(total_rewards, axis=0)
        total_log_probs = torch.stack(total_log_probs, dim=0)
        total_log_prob_masks = torch.stack(total_log_prob_masks, dim=0)

        dis_rewards = []
        gamma = 0.99
        num_samples = total_log_probs.size(1)
        rewards = total_rewards[self.steps_per_level:]
        log_probs = total_log_probs[:-self.steps_per_level]
        log_prob_masks = total_log_prob_masks[:-self.steps_per_level]

        R = np.zeros(num_samples)
        for r in rewards[::-1]:
            R = r + gamma * R
            dis_rewards.insert(0, R)
        dis_rewards = torch.from_numpy(np.array(dis_rewards)).to(log_probs.device)
        policy_loss = dis_rewards * (-1 * log_probs * log_prob_masks).sum(dim=-1)
        # policy_loss = policy_loss.sum(dim=0) * batch_masks
        # policy_loss = policy_loss.sum() / batch_masks.sum()
        policy_loss = policy_loss.sum() / num_samples

        policy_loss.backward()
        self.optimizer.step_and_update_lr()

        return policy_loss.detach().cpu().numpy()

    def create_timeloop_report(self, program, dir_path):
        fitness = self.cost_model.thread_fun((program, 0))
        stats = self.cost_model.thread_fun((program, 0))
        os.makedirs(dir_path, exist_ok=True)
        columns = ['EDP (uJ cycles)', 'Cycles', 'Energy (pJ)', 'Utilization', 'pJ/Algorithm-Compute', 'pJ/Actual-Compute', 'Area (mm2)'][:len(stats)]

        os.system(f'cp -d -r {os.path.join(self.timeloop_out_config_path, "pool-0")}/* {dir_path}')
        with open(os.path.join(dir_path,'Timeloop.txt'), 'w') as fd:
            value = [f'{v:.5e}' for v in fitness]
            fd.write(f'Achieved Fitness: {value}\n')
            fd.write(f'Statistics\n')
            fd.write(f'{columns}\n')
            fd.write(f'{stats}')
        stats = np.array(stats).reshape(1, -1)
        df = pd.DataFrame(stats, columns=columns)
        df.to_csv(os.path.join(dir_path,'Timeloop.csv'))

    def record_chkpt(self, write=False):
        self.best_obj_record.append(self.best_obj)
        self.best_latency_record.append(self.best_latency)
        self.best_energy_record.append(self.best_energy)
        self.best_program_record.append(self.best_program)

        chkpt = None
        if write:
            with open(os.path.join(self.report_dir, 'env_chkpt.plt'), 'wb') as fd:
                chkpt = {
                    'best_obj_record': self.best_obj_record,
                    'best_latency_record': self.best_latency_record,
                    'best_energy_record': self.best_energy_record,
                    'best_program_record': self.best_program_record,
                    'best_obj': self.best_obj,
                    'best_latency': self.best_latency,
                    'best_energy': self.best_energy,
                    'best_program': self.best_program
                }
                pickle.dump(chkpt, fd)
        return chkpt

    def clean_timeloop_output_files(self):
        shutil.rmtree(self.timeloop_out_config_path)
        out_prefix = "./timeloop-model."
        output_file_names = []
        output_file_names.append( "tmp-accelergy.yaml")
        output_file_names.append(out_prefix + "accelergy.log")
        output_file_names.extend(glob.glob("*accelergy.log"))
        output_file_names.extend(glob.glob("*tmp-accelergy.yaml"))
        output_file_names.append(out_prefix + ".log")
        output_file_names.append(out_prefix + "ART.yaml")
        output_file_names.append(out_prefix + "ART_summary.yaml")
        output_file_names.append(out_prefix + "ERT.yaml")
        output_file_names.append(out_prefix + "ERT_summary.yaml")
        output_file_names.append(out_prefix + "flattened_architecture.yaml")
        output_file_names.append(out_prefix + "map+stats.xml")
        output_file_names.append(out_prefix + "map.txt")
        output_file_names.append(out_prefix + "stats.txt")
        for f in output_file_names:
            if os.path.exists(f):
                os.remove(f)





