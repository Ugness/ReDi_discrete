import numpy as np
import torch
import os
from glob import glob

from collections import defaultdict
from math import log

from argparse import ArgumentParser
import json

def entropy_from_counts(counts, base=2):
    """
    Compute the empirical entropy H(X) from a dictionary of counts,
    using either base-2 (bits) or base-e (nats).
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            # Use math.log(p, base) for bits (base=2) or nats (base=np.e)
            entropy -= p * log(p, base)
    return entropy

def compute_total_correlation_x1(x1_group):
    """
    Given an array x1_group of shape (N, 16, 16), where each row is
    one sample of x1, estimate T(X1) = sum_j H(X1_j) - H(X1)
    by naive frequency counting.

    We treat each of the 256 pixels in the 16x16 image as a separate
    discrete variable. We compute:
       H(X1)   via joint frequencies over all 256 pixels
       H(X1_j) via marginal frequencies of each pixel j
    Then T(X1) = sum_j H(X1_j) - H(X1).
    """
    N = x1_group.shape[0]
    if N < 2:
        # With <2 samples, you cannot really estimate correlation reliably.
        # We return 0.0 by convention or skip it entirely.
        return 0.0
    
    # Flatten each image into 256-dimensional vector:
    # shape becomes (N, 256)
    flattened = x1_group.reshape(N, -1)
    d = flattened.shape[1]  # should be 256

    #---- Joint distribution (all 256 dims) ----#
    # We'll store each flattened row as a tuple, then count frequencies.
    joint_counts = defaultdict(int)
    for row in flattened:
        key = tuple(row)   # row is length-256
        joint_counts[key] += 1
    H_joint = entropy_from_counts(joint_counts, base=2)

    #---- Marginal distributions (one pixel at a time) ----#
    # We'll compute an entropy for each pixel dimension j.
    marginal_entropies = []
    for j in range(d):
        counts_j = defaultdict(int)
        for row in flattened:
            val_j = row[j]
            counts_j[val_j] += 1
        H_j = entropy_from_counts(counts_j, base=2)
        marginal_entropies.append(H_j)

    sum_marginals = sum(marginal_entropies)
    
    #---- Total correlation ----#
    T_val = sum_marginals - H_joint
    return T_val, sum_marginals, H_joint

def compute_conditional_total_correlation_x1_given_x0y(x0, x1, y):
    """
    1) Group all samples by unique (x0[i], y[i]) pair.
       - Here x0[i] is a (16, 16) array, and y[i] is a scalar.
    2) Collect the corresponding x1[i] arrays for each group.
    3) For each group, estimate T(X1) by naive frequency counting
       (thus approximating T(X1 | x0=x0_val, y=y_val)).

    Returns a dict:  (x0_bytes, y_val) -> estimated total correlation in bits.
    """

    # Safety checks:
    assert len(x0) == len(x1) == len(y), "All must have same length"
    N = len(x0)
    
    # Group x1 by unique (x0, y)
    groups = defaultdict(list)
    for i in range(N):
        # We need a hashable key for x0[i], which is shape (16,16)
        # Convert to bytes, or a tuple if you prefer
        x0_key = x0[i].tobytes()
        key = (x0_key, y[i])
        groups[key].append(x1[i])

    # Compute total correlation for each group
    results = {}
    marginals = {}
    joints = {}
    for key, x1_list in groups.items():
        # x1_list is a list of arrays, shape each (16,16)
        # print(len(x1_list))
        x1_group = np.stack(x1_list)  # shape = (num_samples_for_this_group, 16, 16)
        T_val, sum_marginals, H_joint = compute_total_correlation_x1(x1_group)
        results[key] = T_val
        marginals[key] = sum_marginals
        joints[key] = H_joint

    return results, joints, marginals


# Suppose you have stored x0.npy, x1.npy, y.npy
# Each x0[i], x1[i] is shape (16,16), and y[i] is shape ()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("base_dir", type=str, default=None, help="Base directory of the dataset")
    parser.add_argument('--force', action='store_true', help='Force recompute even if files exist')
    args = parser.parse_args()
    base_dir = args.base_dir
    force = args.force

    # save as json
    run_id = base_dir.split('dataset/')[-1].split('lambda1.0_grad2')[0].split('_d10')[0]+'_50k'
    sched_mode = 'arccos'
    step = int(base_dir.split('step=')[-1].split('_')[0])
    sm_temp = float(base_dir.split('temp=')[:2][-1].split('_')[0])
    if 'rtemp' in base_dir:
        r_temp = float(base_dir.split('rtemp=')[-1].split('_')[0])
    else:
        r_temp = 4.5
    cfg_w = float(base_dir.split('w=')[-1].split('_')[0])

    fname = f"metrics_{sched_mode}_step={step}_temp={sm_temp}" \
                    f"_rtemp={r_temp}_w={cfg_w}_randomize=linear_{run_id}.json"
    json_path = os.path.join('results', run_id, 'tc_stat',  fname)
    if os.path.exists(json_path) and not force:
        print(f"File {json_path} already exists. Use --force to recompute.")
        exit(0)

    # Compute TC
    x0_arr = np.load(os.path.join(base_dir, "x0.npy"))  # shape (50000,16,16)
    x1_arr = np.load(os.path.join(base_dir, "x1.npy"))  # shape (50000,16,16)
    y_arr  = np.load(os.path.join(base_dir, "y.npy"))   # shape (50000,)

    print(f"x0_arr shape: {x0_arr.shape}")
    if x0_arr.shape[0] != x1_arr.shape[0]:
        x1_arr = x1_arr.reshape(x0_arr.shape[0], 16, 16)
    print(f"x1_arr shape: {x1_arr.shape}")
    print(f"y_arr shape: {y_arr.shape}")

    # Measure the total correlation of x1 given each unique (x0, y).
    tc_results, joints, marginals = compute_conditional_total_correlation_x1_given_x0y(x0_arr, x1_arr, y_arr)

    tc_all = np.array(list(tc_results.values()))
    joints_all = np.array(list(joints.values()))
    marginals_all = np.array(list(marginals.values()))

    avg_tc = np.mean(list(tc_results.values()))
    avg_joints = np.mean(list(joints.values()))
    avg_marginals = np.mean(list(marginals.values()))

    result_dict = {
        'avg_tc': avg_tc,
        'avg_joints': avg_joints,
        'avg_marginals': avg_marginals,
        'sched_mode': sched_mode,
        'step': step,
        'sm_temp': sm_temp,
        'cfg_w': cfg_w,
        'r_temp': r_temp,
    }

    json_str = json.dumps(result_dict, indent=4)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        f.write(json_str)
    
    # save tc_all, joints_all, marginals_all as npy files
    np.save(os.path.join('results', run_id, 'tc_stat', 'tc_all.npy'), tc_all)
    np.save(os.path.join('results', run_id, 'tc_stat', 'joints_all.npy'), joints_all)
    np.save(os.path.join('results', run_id, 'tc_stat', 'marginals_all.npy'), marginals_all)

    print(f"Saved results to {json_path}")
    print(f"Saved tc_all, joints_all, marginals_all to npy files")

    print(f"Run ID: {run_id}")
    print(f"Average total correlation: {avg_tc}")
    print(f"Average joint entropy: {avg_joints}")
    print(f"Average marginal entropy: {avg_marginals}")