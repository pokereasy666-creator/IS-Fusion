_base_ = ['./isfusion_0075voxel.py']

# A30 (24 GB) — full-resolution model, larger batch size
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

# Scale lr linearly with effective batch size:
# base: 0.00000625 for 1 GPU × 1 sample  →  0.0000125 for 1 GPU × 2 samples
optimizer = dict(lr=0.0000125)
