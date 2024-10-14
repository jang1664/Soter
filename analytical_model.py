import math

from torch.distributions import Categorical
import numpy as np
import copy
import torch
import torch.nn.functional as F


class Model(object):
    def __init__(self, prime2idx, buffer_size_list, buf_spmap_cstr, steps_per_level, operator_instance, accelerator):
        super().__init__()

        self.prime2idx = prime2idx
        self.idx2prime = {value: key for key, value in prime2idx.items()}
        self.num_primes = len(self.prime2idx.keys())

        self.buffer_size_list = buffer_size_list
        self.buf_spmap_cstr = buf_spmap_cstr
        self.steps_per_level = steps_per_level
        self.operator_instance = operator_instance
        self.accelerator = accelerator

    def get_remain_buffer_size(self, cur_buffer_level, program_seq_disorder, loop_ind):
        if 'Simba' in self.accelerator:
            return self.get_remain_buffer_size_for_simba(cur_buffer_level, program_seq_disorder, loop_ind)
        elif 'Eyeriss' in self.accelerator:
            return self.get_remain_buffer_size_for_eyeriss(cur_buffer_level, program_seq_disorder, loop_ind)
        elif 'TensorCore' in self.accelerator:
            return self.get_remain_buffer_size_for_tensorcore(cur_buffer_level, program_seq_disorder, loop_ind)


    def get_remain_buffer_size_for_simba(self, cur_buffer_level, program_seq_disorder, loop_ind):
        """
          Calculate the remaining buffer size for current buffer level
          buffer level one is register, last one is dram

          Returns
          ------
          remain_buffer_size:
            remaining buffer size for current buffer level. this will be compared with later buffer size and selected less one.
            later buffer level -> to Dram level. so later one sholud have more large remain buffer size.
        """

        # calculate tile size for each dimension
        # start from lowest level buffer to one previous level from current level
        # exploration start from lowest buffer level, so previous tile sizes are already decided, so no problem
        # but current buffer size can be calculated when current buffer factor is decided, so assume minimum factor for not decided ones
        buffer_size = self.buffer_size_list[f'l{cur_buffer_level}']
        batch_size = program_seq_disorder.size(0)
        tiles = program_seq_disorder.new_ones(batch_size, self.steps_per_level)
        for buffer_idx in range(1, cur_buffer_level + 1):
            start_ind = (buffer_idx - 1) * self.steps_per_level
            end_ind = buffer_idx * self.steps_per_level
            level_program_seq_disorder = copy.deepcopy(program_seq_disorder[:, start_ind:end_ind])
            for k, v in self.prime2idx.items():
                tiles *= torch.pow(int(k), level_program_seq_disorder[:, :, v + 1])

        # split the tiles into individual dimensions. each variable is 1D vector with batch dimension
        R, S, P, Q, C, K, H, N = torch.unbind(tiles, dim=1)
        wstride = self.operator_instance['Wstride']
        hstride = self.operator_instance['Hstride']
        wdilation = self.operator_instance['Wdilation']
        hdilation = self.operator_instance['Hdilation']
        if cur_buffer_level == 1 or cur_buffer_level == 3:  # pe reg/weight
            N = program_seq_disorder.new_zeros(batch_size)
            P = program_seq_disorder.new_zeros(batch_size)
            Q = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 2:  # pe acc
            C = program_seq_disorder.new_zeros(batch_size)
            if self.operator_instance['type'] == 'C2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 4:  # pe input
            K = program_seq_disorder.new_zeros(batch_size)
            if self.operator_instance['type'] == 'T2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)

        # we will calculate how much tiles of previous buffer level can be stored in current buffer level
        # when we consider M, K and N factor of current buffer level set at minimum. so weight tile size of current level is same as previous level.
        # As a result, the number of tiles of previous buffer for input(MK) and output(MN) can be calculated by subtracting weight tile size from buffer size and 
        # divided by (MK + MN).
        # If remain buffer size is 5, so we can expand M dimension up to 5 times when assuming K and N are minimum.
        #TODO: why current level factors that are selected already are not considered in this calculation? we don't need to assum already selected factors as minimum.

        # default expression
        if self.operator_instance['type'] == 'C2D':
            # tile size for next buffer level
            input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            weight_tile = K * R * S * C * H
            output_tile = P * Q * K * N * H

            # calculate current buffer tile size when selecting loop_ind
            # <dim_name>_sub / <dim_name>_coef is the tile size for current buffer level
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
            K_coef = (R * S * C + P * Q * N) * K * H
            C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S * H
            H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * R * S * C + P * Q * K * N) * H
        elif self.operator_instance['type'] == 'T2D':
            input_tile = P * Q * C * N * H
            weight_tile = K * R * S * C * H
            output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
            K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + C * R * S) * K * H
            C_coef = (P * Q * N + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + K * R * C) * S * H
            H_coef = (P * Q * C * N + K * R * S * C + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H

        # for global buffer we should recalculate the tile sizes
        if cur_buffer_level == 5:  # global buffer
            if self.operator_instance['type'] == 'C2D':
                input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                weight_tile = program_seq_disorder.new_zeros(batch_size)
                output_tile = P * Q * K * N * H
                N_sub = weight_tile # KN
                K_sub = input_tile # MK
                C_sub = output_tile # MN

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H # MK + MN
                K_coef = P * Q * N * K * H # MN
                C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * C * H # MK
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C) * S * H
                H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K * N) * H
            elif self.operator_instance['type'] == 'T2D':
                input_tile = P * Q * C * N * H
                weight_tile = program_seq_disorder.new_zeros(batch_size)
                output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
                R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
                K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * K * H
                C_coef = (P * Q * N) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K) * S * H
                H_coef = (P * Q * C * N + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        else:
            # disable some dimension for some buffer levels
            if cur_buffer_level == 1 or cur_buffer_level == 3:  # pe/reg weight
                # N, P, Q is unrelated with weight. So for N, P, Q, no buffer size reduction.
                N = program_seq_disorder.new_zeros(batch_size)
                P = program_seq_disorder.new_zeros(batch_size)
                Q = program_seq_disorder.new_zeros(batch_size)

                N_sub = program_seq_disorder.new_zeros(batch_size).float()
                P_sub = program_seq_disorder.new_zeros(batch_size).float()
                Q_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                P_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                Q_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 2:  # pe acc
                C = program_seq_disorder.new_zeros(batch_size)
                C_sub = program_seq_disorder.new_zeros(batch_size).float()
                C_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.operator_instance['type'] == 'C2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

            elif cur_buffer_level == 4:  # pe input
                K = program_seq_disorder.new_zeros(batch_size)
                K_sub = program_seq_disorder.new_zeros(batch_size).float()
                K_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.operator_instance['type'] == 'T2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

        coef_arr = torch.stack([R_coef, S_coef, P_coef, Q_coef, C_coef, K_coef, H_coef, N_coef], dim=1)[:, loop_ind]
        sub_arr = torch.stack([R_sub, S_sub, P_sub, Q_sub, C_sub, K_sub, H_sub, N_sub], dim=1)[:, loop_ind]

        remain_buffer_size = (buffer_size - sub_arr.float()) / coef_arr.float()

        return remain_buffer_size

    def get_remain_buffer_size_for_eyeriss(self, cur_buffer_level, program_seq_disorder, loop_ind):
        buffer_size = self.buffer_size_list[f'l{cur_buffer_level}']
        batch_size = program_seq_disorder.size(0)
        tiles = program_seq_disorder.new_ones(batch_size, self.steps_per_level)
        for buffer_idx in range(1, cur_buffer_level + 1):
            start_ind = (buffer_idx - 1) * self.steps_per_level
            end_ind = buffer_idx * self.steps_per_level
            level_program_seq_disorder = copy.deepcopy(program_seq_disorder[:, start_ind:end_ind])
            for k, v in self.prime2idx.items():
                tiles *= torch.pow(int(k), level_program_seq_disorder[:, :, v + 1])

        R, S, P, Q, C, K, H, N = torch.unbind(tiles, dim=1)
        wstride = self.operator_instance['Wstride']
        hstride = self.operator_instance['Hstride']
        wdilation = self.operator_instance['Wdilation']
        hdilation = self.operator_instance['Hdilation']
        if cur_buffer_level == 1:  # pe acc
            C = program_seq_disorder.new_zeros(batch_size)
            if self.operator_instance['type'] == 'C2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 2:  # pe weight
            N = program_seq_disorder.new_zeros(batch_size)
            P = program_seq_disorder.new_zeros(batch_size)
            Q = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 3:  # pe input
            K = program_seq_disorder.new_zeros(batch_size)
            if self.operator_instance['type'] == 'T2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 4:  # dummybuffer
            H = program_seq_disorder.new_zeros(batch_size)
            N = program_seq_disorder.new_zeros(batch_size)
            K = program_seq_disorder.new_zeros(batch_size)
            C = program_seq_disorder.new_zeros(batch_size)
            P = program_seq_disorder.new_zeros(batch_size)
            Q = program_seq_disorder.new_zeros(batch_size)
            R = program_seq_disorder.new_zeros(batch_size)
            S = program_seq_disorder.new_zeros(batch_size)

        if self.operator_instance['type'] == 'C2D':
            input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            weight_tile = K * R * S * C * H
            output_tile = P * Q * K * N * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
            K_coef = (R * S * C + P * Q * N) * K * H
            C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S * H
            H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * R * S * C + P * Q * K * N) * H
        elif self.operator_instance['type'] == 'T2D':
            input_tile = P * Q * C * N * H
            weight_tile = K * R * S * C * H
            output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
            K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + C * R * S) * K * H
            C_coef = (P * Q * N + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + K * R * C) * S * H
            H_coef = (P * Q * C * N + K * R * S * C + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        if cur_buffer_level == 5:  # global buffer
            if self.operator_instance['type'] == 'C2D':
                input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                weight_tile = program_seq_disorder.new_zeros(batch_size)
                output_tile = P * Q * K * N * H
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
                K_coef = P * Q * N * K * H
                C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C) * S * H
                H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K * N) * H
            elif self.operator_instance['type'] == 'T2D':
                input_tile = P * Q * C * N * H
                weight_tile = program_seq_disorder.new_zeros(batch_size)
                output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
                R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
                K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * K * H
                C_coef = (P * Q * N) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K) * S * H
                H_coef = (P * Q * C * N + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        else:
            if cur_buffer_level == 1:  # pe acc
                C = program_seq_disorder.new_zeros(batch_size)
                C_sub = program_seq_disorder.new_zeros(batch_size).float()
                C_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.operator_instance['type'] == 'C2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 2:  # pe weight
                N = program_seq_disorder.new_zeros(batch_size)
                P = program_seq_disorder.new_zeros(batch_size)
                Q = program_seq_disorder.new_zeros(batch_size)
                N_sub = program_seq_disorder.new_zeros(batch_size).float()
                P_sub = program_seq_disorder.new_zeros(batch_size).float()
                Q_sub = program_seq_disorder.new_zeros(batch_size).float()
                N_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                P_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                Q_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 3:  # pe input
                K = program_seq_disorder.new_zeros(batch_size)
                K_sub = program_seq_disorder.new_zeros(batch_size).float()
                K_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.operator_instance['type'] == 'T2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 4:     # dummybuffer
                H_sub = program_seq_disorder.new_zeros(batch_size).float()
                N_sub = program_seq_disorder.new_zeros(batch_size).float()
                K_sub = program_seq_disorder.new_zeros(batch_size).float()
                C_sub = program_seq_disorder.new_zeros(batch_size).float()
                P_sub = program_seq_disorder.new_zeros(batch_size).float()
                Q_sub = program_seq_disorder.new_zeros(batch_size).float()
                R_sub = program_seq_disorder.new_zeros(batch_size).float()
                S_sub = program_seq_disorder.new_zeros(batch_size).float()
                H_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                N_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                K_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                C_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                P_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                Q_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

        coef_arr = torch.stack([R_coef, S_coef, P_coef, Q_coef, C_coef, K_coef, H_coef, N_coef], dim=1)[:, loop_ind]
        sub_arr = torch.stack([R_sub, S_sub, P_sub, Q_sub, C_sub, K_sub, H_sub, N_sub], dim=1)[:, loop_ind]

        remain_buffer_size = (buffer_size - sub_arr.float()) / coef_arr.float()

        return remain_buffer_size

    def get_remain_buffer_size_for_tensorcore(self, cur_buffer_level, program_seq_disorder, loop_ind):
        buffer_size = self.buffer_size_list[f'l{cur_buffer_level}']
        batch_size = program_seq_disorder.size(0)
        tiles = program_seq_disorder.new_ones(batch_size, self.steps_per_level)
        for buffer_idx in range(1, cur_buffer_level + 1):
            start_ind = (buffer_idx - 1) * self.steps_per_level
            end_ind = buffer_idx * self.steps_per_level
            level_program_seq_disorder = copy.deepcopy(program_seq_disorder[:, start_ind:end_ind])
            for k, v in self.prime2idx.items():
                tiles *= torch.pow(int(k), level_program_seq_disorder[:, :, v + 1])

        R, S, P, Q, C, K, H, N = torch.unbind(tiles, dim=1)
        wstride = self.operator_instance['Wstride']
        hstride = self.operator_instance['Hstride']
        wdilation = self.operator_instance['Wdilation']
        hdilation = self.operator_instance['Hdilation']
        if cur_buffer_level == 1:  # LRF
            K = program_seq_disorder.new_zeros(batch_size)
            if self.operator_instance['type'] == 'T2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 2:  # RF
            C = program_seq_disorder.new_zeros(batch_size)
            if self.operator_instance['type'] == 'C2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)

        if self.operator_instance['type'] == 'C2D':
            input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            weight_tile = K * R * S * C * H
            output_tile = P * Q * K * N * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
            K_coef = (R * S * C + P * Q * N) * K * H
            C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S * H
            H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * R * S * C + P * Q * K * N) * H
        elif self.operator_instance['type'] == 'T2D':
            input_tile = P * Q * C * N * H
            weight_tile = K * R * S * C * H
            output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
            K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + C * R * S) * K * H
            C_coef = (P * Q * N + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + K * R * C) * S * H
            H_coef = (P * Q * C * N + K * R * S * C + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        if cur_buffer_level == 3:  # SMEM
            if self.operator_instance['type'] == 'C2D':
                input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                weight_tile = K * R * S * C * H
                output_tile = program_seq_disorder.new_zeros(batch_size)
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * N * H
                K_coef = (R * S * C) * K * H
                C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S * H
                H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * R * S * C) * H
            elif self.operator_instance['type'] == 'T2D':
                input_tile = P * Q * C * N * H
                weight_tile = K * R * S * C * H
                output_tile = program_seq_disorder.new_zeros(batch_size)
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile
                Q_sub = weight_tile
                R_sub = input_tile
                S_sub = input_tile
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = P * Q * C * N * H
                K_coef = C * R * S * K * H
                C_coef = (P * Q * N + K * R * S) * C * H
                P_coef = Q * C * N * P * H
                Q_coef = P * C * N * Q * H
                R_coef = K * S * C * R * H
                S_coef = K * R * C * S * H
                H_coef = (P * Q * C * N + K * R * S * C) * H
        else:
            if cur_buffer_level == 1:  # LRF
                K = program_seq_disorder.new_zeros(batch_size)
                K_sub = program_seq_disorder.new_zeros(batch_size).float()
                K_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.operator_instance['type'] == 'T2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 2:  # pe acc
                C = program_seq_disorder.new_zeros(batch_size)
                C_sub = program_seq_disorder.new_zeros(batch_size).float()
                C_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.operator_instance['type'] == 'C2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

        coef_arr = torch.stack([R_coef, S_coef, P_coef, Q_coef, C_coef, K_coef, H_coef, N_coef], dim=1)[:, loop_ind]
        sub_arr = torch.stack([R_sub, S_sub, P_sub, Q_sub, C_sub, K_sub, H_sub, N_sub], dim=1)[:, loop_ind]

        remain_buffer_size = (buffer_size - sub_arr.float()) / coef_arr.float()

        return remain_buffer_size
    def get_remain_buf_spmap(self, cur_buffer_level, program_seq_disorder):
        """
        calculate how many factor of 2 can be used for spatial.
        If L2 buffer instance is 2 and L3 buffer instance 16, total spatial tile budget is 8.
        If previous step use factor of 2, remain factor is 4.
        we iterate previous steps and calculate how many factor of 2 can be used for spatial.
        """
        buf_spmap_cstr = self.buf_spmap_cstr[f'l{cur_buffer_level}']
        start_ind = (cur_buffer_level - 1) * self.steps_per_level # first dim for current buffer level
        end_ind = cur_buffer_level * self.steps_per_level
        level_program_seq_disorder = copy.deepcopy(program_seq_disorder[:, start_ind:end_ind])
        num_samples = program_seq_disorder.size(0)
        used_buf_spmap = program_seq_disorder.new_ones(num_samples)
        # print(buf_spmap_cstr)
        for i in range(self.steps_per_level):
            sp_tile2 = level_program_seq_disorder[:, i, self.num_primes + 1]
            used_buf_spmap *= torch.clamp(torch.pow(2, sp_tile2), min=1)
        remain_buf_spmap = buf_spmap_cstr / used_buf_spmap.float()

        return remain_buf_spmap

    def get_max_temporal_size(self, cur_buffer_level, finished_levels, tile2_remain_dimension_budgets, remain_buf_spmap):
        """
        we calculate maximum temporal factor of current buffer level.
        spatial factor only use factor of 2. and it is good to consume all spatial factor budget.
        So constraint temporal factor to remaining amount by subtracting spatial factor.
        Spatial factor can be used across all dimensions, so we first sum factor of 2 for all dimensions and subtract 
        allowed spatial factors from unfinished buffer level
        """
        max_temporal_tile2 = tile2_remain_dimension_budgets - torch.log2(torch.clamp(remain_buf_spmap, min=1))

        for level in range(1, len(self.buffer_size_list) + 1):
            buf_spmap_cstr = self.buf_spmap_cstr[f'l{level}']
            if level not in finished_levels and level != cur_buffer_level:
                max_temporal_tile2 -= math.log2(buf_spmap_cstr)
        return torch.clamp(max_temporal_tile2, min=0).long()

    def get_spatial_size(self, mode, tile2_max, loop_ind,
                         tile2_remain_budget, remain_buf_spmap):
        """
          calculate spatial factor of 2 for current buffer level.
          We calculated maximum spatial factor from instance number relation. But we have maximum factor 2 of each dim.
          So we minimize spatial factor of 2 by considering remaining factor 2 count.

          For minimum, if current level is last, set min to max, so no remain spatial factor.
          Otherwise, calculate remaining factor 2 for other dimensions and subtract them from spatial factor.
          If the result is more than one, set min to the result, so we ensure spatial factor is consumed exaustivly
        """
        sp_tile2_max = torch.log2(torch.clamp(remain_buf_spmap, min=1))
        sp_tile2_max = torch.clamp(sp_tile2_max.long(), min=0)
        sp_tile2_max = torch.minimum(sp_tile2_max, tile2_max)

        if mode % self.steps_per_level == self.steps_per_level - 1:
            sp_tile2_min = sp_tile2_max
        else:
            tile2_remain_dimension_budgets = (tile2_remain_budget[:, loop_ind + 1:]).sum(dim=-1)
            sp_tile2_min = torch.clamp(
                torch.log2(torch.clamp(remain_buf_spmap, min=1)) - tile2_remain_dimension_budgets, min=0)
            sp_tile2_min = torch.minimum(sp_tile2_min.long(), sp_tile2_max)
        return sp_tile2_max, sp_tile2_min

