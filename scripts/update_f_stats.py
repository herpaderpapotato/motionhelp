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
n_frames = 0


KP_FILE_MULTICLASS = "keypoints/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
EMB_FILE_MULTICLASS = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"

embeddings_found = 0
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
np.savez('data/feature_stats.npz', **stats)
print('Saved data/feature_stats.npz')