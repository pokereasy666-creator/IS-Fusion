_base_ = ['./isfusion_0075voxel.py']

# A30 (24 GB) — with fp16 + gradient checkpointing, batch 1 uses ~9.4 GB.
# Increase to batch 2 to better utilize the 24 GB VRAM.
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

# Scale learning rate linearly with batch size (2x batch → 2x lr)
optimizer = dict(lr=1.25e-05)
