import pickle
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import sys
from copy import copy, deepcopy
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go


sys.path.append("/root/project/soter_v2")
sys.path.append("/root/project/soter_v2/metric_space_analysis")
sys.path.append("/root/project/soter_v2/basic")

from metric_space_analysis.search import *
from basic.loop_topo import *
from basic.arch import *
from basic.problem import *
import search

"""
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
record["map_record"]["epoch"].append(ep)
record["map_record"]["map"].append(map)
record["map_record"]["batch_idx"].append(i)
record["map_record"]["cycle"].append(latency[i])
record["map_record"]["energy"].append(energy[i])
record["map_record"]["edp"].append(latency[i] * energy[i])
"""

def doAnalysis(record_path, arch_path, report_postfix):
  print(f"Analysis: {record_path} {arch_path}")
  # arch_name = os.path.basename(arch_path).split(".")[0]
  # record_name = os.path.basename(record_path).split(".")[0]
  arch_name = arch_path.split("/")[-2]
  record_name = record_path.split("/")[-2]
  output_path = f"{arch_name}_{record_name}_{report_postfix}"
  os.makedirs(output_path, exist_ok=True)

  fd = open(f"{output_path}/{arch_name}_{record_name}.log", "w")
  # data = pickle.load(open("report/arch_Simba/obj_edp/bertlarge_input1/layer-4/record.pkl", "rb"))
  # data = pickle.load(open("report/arch_Simba/obj_edp/bertlarge_input1/layer-4/long_search/record.pkl", "rb"))
  # data = pickle.load(open("report/arch_Simba/obj_edp/bertlarge_input1/layer-4/long_search_random_search/record.pkl", "rb"))
  data = pickle.load(open(record_path, "rb"))

  # arch = Arch(yaml.load(open("/root/project/Soter/SpatialAccelerators/Simba/arch.yaml", "r"), Loader=yaml.FullLoader))
  arch = Arch(yaml.load(open(arch_path, "r"), Loader=yaml.FullLoader))

  prob_desc = ProblemDesc(ProblemDesc.convertToMKN(data["problem"]))

  #insert loop topology column
  map_records = data["map_record"]
  new_map_records = deepcopy(map_records)
  new_map_records["topo"] = []
  for map_data in new_map_records["map"]:
    topo = LoopTopo(arch, prob_desc, map_data)
    new_map_records["topo"].append(topo)

  #	epoch	map	batch_idx	cycle	energy	edp	topo
  df = pd.DataFrame(new_map_records)

  #cycle	energy	edp
  #min	max	mean	min	max	mean	min	max	mean
  #epoch	
  stat_epoch_df = df.groupby('epoch').agg({
      'cycle': ['min', 'max', 'mean'],
      'energy': ['min', 'max', 'mean'],
      'edp': ['min', 'max', 'mean']
  })

  # group by topo
  ## num of search
  ## min, mean, max, std
  df_by_topo = df.groupby('topo').agg(
    count=('topo', 'count'),
    cycle=('cycle', 'mean'),
    energy=('energy', 'mean'),
    edp=('edp', 'mean')
  )

  fd.write(f"df_by_topo length is {len(df_by_topo)}\n")

  plt.figure(figsize=(10, 6))
  # plt.plot(stat_epoch_df.index, stat_epoch_df[('cycle', 'min')], label='Min Cycles')
  plt.plot(stat_epoch_df.index, stat_epoch_df[('cycle', 'mean')], label='Max Cycles')
  plt.xlabel('Epoch')
  plt.ylabel('Cycles')
  plt.legend()
  plt.savefig(f"{output_path}/{arch_name}_{record_name}_cycle.png")
  plt.close()

  plt.figure(figsize=(10, 6))
  # plt.plot(stat_epoch_df.index, stat_epoch_df[('cycle', 'min')], label='Min Cycles')
  plt.plot(stat_epoch_df.index, stat_epoch_df[('energy', 'mean')], label='energy')
  plt.xlabel('Epoch')
  plt.ylabel('energy')
  plt.legend()
  plt.savefig(f"{output_path}/{arch_name}_{record_name}_energy.png")
  plt.close()

  plt.figure(figsize=(10, 6))
  plt.plot(stat_epoch_df.index, stat_epoch_df[('edp', 'mean')], label='edp')
  plt.legend()
  plt.savefig(f"{output_path}/{arch_name}_{record_name}_edp.png")
  plt.close()

  min_cycle_map = df.loc[df['cycle'].idxmax()]
  min_enery_map = df.loc[df['energy'].idxmax()]
  min_edp_map = df.loc[df['edp'].idxmin()]

  yaml.dump(min_cycle_map["map"], open(f"{output_path}/min_cycle_map.yaml", "w"))
  yaml.dump(min_enery_map["map"], open(f"{output_path}/min_energy_map.yaml", "w"))
  yaml.dump(min_edp_map["map"], open(f"{output_path}/min_edp_map.yaml", "w"))

  fd.write(f"min_cycle_map: {min_cycle_map}\n")
  fd.write(f"min_enery_map: {min_enery_map}\n")
  fd.write(f"min_edp_map: {min_edp_map}\n")

def doPCAAnalysis(record_path, arch_path, postfix):
  data = pickle.load(open(record_path, "rb"))
  arch_name = arch_path.split("/")[-2]
  record_name = record_path.split("/")[-2]
  output_path = f"{arch_name}_{record_name}_{postfix}"
  os.makedirs(output_path, exist_ok=True)
  print(f"PCA Analysis: {record_path} {arch_path}")

  arch = Arch(yaml.load(open(arch_path, "r"), Loader=yaml.FullLoader))
  prob_desc = ProblemDesc(ProblemDesc.convertToMKN(data["problem"]))

  energys = []
  cycles = []
  edps = []
  factors = []
  for map_inst, energy, cycle in zip(data["map_record"]["map"], data["map_record"]["energy"], data["map_record"]["cycle"]):
    topo = LoopTopo(arch, prob_desc, LoopTopo.convertToMKN(map_inst))
    vector = topo.getProgramVector()
    factors.append(vector)
    energys.append(-energy)
    cycles.append(-cycle)
    edps.append(energy * cycle)

  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(np.array(factors))

  for name, data in zip(['energy', 'cycle', 'edp'], [energys, cycles, edps]):
    plt.figure(figsize=(5,5),constrained_layout=True)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data, cmap='viridis', s=1, alpha=0.7)
    plt.colorbar(label='y')
    plt.savefig(f"{output_path}/{name}_pca.png")
    plt.close()

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], data, c=data, cmap='viridis', alpha=0.7)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel(name)
    plt.savefig(f"{output_path}/{name}_pca_3d.png")
    plt.close()

    # Create a scatter plot
    scatter = go.Scatter3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=data,
        mode='markers',
        marker=dict(
            size=5,
            color=data,  # Color by Z values
            colorscale='Viridis',  # Choose a colorscale
            opacity=0.8
        )
    )

    # Set up the layout for the plot
    layout = go.Layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        title='Interactive 3D Scatter Plot'
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.write_html(f"{output_path}/{name}_pca_3d.html")

if __name__ == "__main__":
  doAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-0/ep100/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_0")
  doAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-0/ep100_random/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_0")

  doAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-3/ep50/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_3")
  doAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-3/ep50_random/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_3")

  doAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-4/ep50/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_4")
  doAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-4/ep50_random/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_4")

  doAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-5/ep50/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_5")
  doAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-5/ep50_random/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_5")

  # PCA
  doPCAAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-0/ep100/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_0")
  doPCAAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-0/ep100_random/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_0")

  doPCAAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-3/ep50/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_3")
  doPCAAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-3/ep50_random/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_3")

  doPCAAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-4/ep50/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_4")
  doPCAAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-4/ep50_random/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_4")

  doPCAAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-5/ep50/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_5")
  doPCAAnalysis("../report/arch_TensorCore/obj_edp/bertlarge_input1/layer-5/ep50_random/record.pkl", "../SpatialAccelerators/TensorCore/arch.yaml", "layer_5")