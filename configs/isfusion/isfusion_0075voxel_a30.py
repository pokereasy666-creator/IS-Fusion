_base_ = ['./isfusion_0075voxel.py']

# A30 (24 GB) — with fp16 + gradient checkpointing, batch 2 uses ~15.4 GB.
# Increase to batch 3 to better utilize the 24 GB VRAM (~21-22 GB expected).
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
)

# Scale learning rate linearly with batch size (3x batch → 3x lr)
optim_wrapper = dict(optimizer=dict(lr=1.875e-05))
