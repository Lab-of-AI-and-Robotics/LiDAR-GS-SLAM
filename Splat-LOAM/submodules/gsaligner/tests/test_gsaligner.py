import matplotlib.pyplot as plt
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from gsaligner import GSAligner

BASE_PATH = Path(__file__).parent / "data"

torch.manual_seed(42)

pcd_ref = o3d.io.read_point_cloud(str(BASE_PATH / "cloud_ref.ply"))
pcd_query = o3d.io.read_point_cloud(str(BASE_PATH / "cloud_query.ply"))
np_depth_ref = np.load(BASE_PATH / "depth_ref.npy")
np_depth_query = np.load(BASE_PATH / "depth_query.npy")
np_projmat = np.load(BASE_PATH / "K_ref.npy")

depth_ref = torch.from_numpy(np_depth_ref).unsqueeze(0).float().cuda()
depth_query = torch.from_numpy(np_depth_query).unsqueeze(0).float().cuda()
projmat = torch.eye(4).float().cuda()
projmat[:3, :3] = torch.from_numpy(np_projmat).float().cuda()
projmat = projmat.transpose(0, 1)

cloud_ref = torch.from_numpy(np.asarray(pcd_ref.points)).float().cuda()
cloud_query = torch.from_numpy(np.asarray(pcd_query.points)).float().cuda()

aligner = GSAligner(height=depth_ref.shape[1],
                    width=depth_ref.shape[2],
                    geom_b_min=0.1,
                    geom_b_max=0.2,
                    geom_b_ratio=0.02,
                    geom_rho_kernel=0.8,
                    geom_iterations=10,
                    photo_omega_depth=5.0,
                    photo_depth_rejection_threshold=0.8,
                    photo_rho_kernel=1.0,
                    photo_min_depth=0.5,
                    photo_max_depth=80.0,
                    photo_iterations=10)

aligner.set_reference(depth_ref, cloud_ref, projmat)
aligner.set_query(depth_query, cloud_query, projmat)


est_T, fitness, inlier_err = aligner.align(torch.eye(4).float().cuda())

print(f"Fitness: {fitness}")
print(f"Inlier error: {inlier_err}")
print(f"Estimated transformation:\n{est_T}")

print("Performance testing")
print("Running 1000 geometric registrations")
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for i in range(1000):
    aligner.align_geometric(torch.eye(4).float().cuda())
end.record()
torch.cuda.synchronize()
print(
    f"Elapsed time: {start.elapsed_time(end):.3f} ms | [{start.elapsed_time(end) / 1000 :.3f} ms per registration]")
print("Running 1000 photometric registrations")
start.record()
for i in range(1000):
    aligner.align_photometric(torch.eye(4).float().cuda())
end.record()
torch.cuda.synchronize()
print(
    f"Elapsed time: {start.elapsed_time(end):.3f} ms [{start.elapsed_time(end) / 1000 :.3f} ms per registration]")
print("Running 1000 combined registrations")
start.record()
for i in range(1000):
    aligner.align(torch.eye(4).float().cuda())
end.record()
torch.cuda.synchronize()
print(
    f"Elapsed time: {start.elapsed_time(end):.3f} ms [{start.elapsed_time(end) / 1000 :.3f} ms per registration]")
