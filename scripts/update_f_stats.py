import numpy as np, os, json

proc = 'data/processed'
scenes = [d for d in sorted(os.listdir(proc)) if os.path.isdir(os.path.join(proc, d))]

base_size = 512
n_people = 6
emb_size = base_size * n_people

# Accumulate running stats
emb_sum = np.zeros(emb_size, dtype=np.float64)
emb_sq_sum = np.zeros(emb_size, dtype=np.float64)
flow_sum = np.zeros(64, dtype=np.float64)
flow_sq_sum = np.zeros(64, dtype=np.float64)
# Dense flow stats: per-channel [2] mean/std
flow_dense_sum = np.zeros(2, dtype=np.float64)
flow_dense_sq_sum = np.zeros(2, dtype=np.float64)
n_dense_pixels = 0
n_frames = 0


KP_FILE_MULTICLASS = "keypoints/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
EMB_FILE_MULTICLASS = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
DENSE_FLOW_FILE = "flow/raft_dense_32x32_s0.5.npy"

embeddings_found = 0
dense_found = 0
for s in scenes:
    d = os.path.join(proc, s)
    if not os.path.exists(os.path.join(d, EMB_FILE_MULTICLASS)):
        continue
    e = np.load(os.path.join(d, EMB_FILE_MULTICLASS)).reshape(-1, emb_size).astype(np.float64)
    embeddings_found += 1
    n = len(e)
    emb_sum += e.sum(axis=0)
    emb_sq_sum += (e**2).sum(axis=0)

    try:
        f = np.load(os.path.join(d, 'flow\\raft_f64_s0.5.npy')).astype(np.float64)
        flow_sum += f.sum(axis=0)
        flow_sq_sum += (f**2).sum(axis=0)
    except FileNotFoundError:
        pass

    # Dense flow stats
    dense_path = os.path.join(d, DENSE_FLOW_FILE)
    if os.path.exists(dense_path):
        df = np.load(dense_path).astype(np.float64)  # [T, 2, H, W]
        dense_found += 1
        # Per-channel stats: mean over T, H, W dimensions
        for ch in range(2):
            ch_data = df[:, ch]  # [T, H, W]
            flow_dense_sum[ch] += ch_data.sum()
            flow_dense_sq_sum[ch] += (ch_data ** 2).sum()
        n_dense_pixels += df.shape[0] * df.shape[2] * df.shape[3]

    n_frames += n

emb_mean = emb_sum / n_frames
emb_std = np.sqrt(emb_sq_sum / n_frames - emb_mean**2)
emb_std = np.maximum(emb_std, 1e-6)

flow_mean = flow_sum / n_frames
flow_std = np.sqrt(flow_sq_sum / n_frames - flow_mean**2)
flow_std = np.maximum(flow_std, 1e-6)

print(f'Computed over {n_frames} frames from {len(scenes)} scenes')
print(f'Emb mean range: [{emb_mean.min():.4f}, {emb_mean.max():.4f}]')
print(f'Emb std range:  [{emb_std.min():.4f}, {emb_std.max():.4f}]')
print(f'Flow mean range: [{flow_mean.min():.4f}, {flow_mean.max():.4f}]')
print(f'Flow std range:  [{flow_std.min():.4f}, {flow_std.max():.4f}]')
print(f'Embeddings found in {embeddings_found}/{len(scenes)} scenes')

# Save stats
stats = {
    'emb_mean': emb_mean.astype(np.float32),
    'emb_std': emb_std.astype(np.float32),
    'flow_mean': flow_mean.astype(np.float32),
    'flow_std': flow_std.astype(np.float32),
    'n_frames': int(n_frames),
}

# Dense flow stats
if n_dense_pixels > 0:
    flow_dense_mean = flow_dense_sum / n_dense_pixels
    flow_dense_std = np.sqrt(flow_dense_sq_sum / n_dense_pixels - flow_dense_mean**2)
    flow_dense_std = np.maximum(flow_dense_std, 1e-6)
    # Store as [2, 1, 1] for broadcasting over [T, 2, H, W]
    stats['flow_dense_mean'] = flow_dense_mean.astype(np.float32).reshape(2, 1, 1)
    stats['flow_dense_std'] = flow_dense_std.astype(np.float32).reshape(2, 1, 1)
    print(f'Dense flow found in {dense_found}/{len(scenes)} scenes ({n_dense_pixels} pixels)')
    print(f'Dense flow mean: [{flow_dense_mean[0]:.6f}, {flow_dense_mean[1]:.6f}]')
    print(f'Dense flow std:  [{flow_dense_std[0]:.6f}, {flow_dense_std[1]:.6f}]')
else:
    print(f'No dense flow files found — skipping dense stats')

np.savez('data/featurestats/feature_stats.npz', **stats)
print('Saved data/featurestats/feature_stats.npz')