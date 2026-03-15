_base_ = ['./isfusion_0075voxel.py']

# A30 (24 GB) — full-resolution model, single sample per GPU
# The full model (Swin-T + SparseEncoder + ISFusion) in fp32 is tight on 24 GB,
# so keep batch size at 1 and rely on fp16 + gradient checkpointing (inherited).
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)
