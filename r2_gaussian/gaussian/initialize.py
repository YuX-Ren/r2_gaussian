import os
import sys
import os.path as osp
import numpy as np

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.arguments import ModelParams
from r2_gaussian.utils.graphics_utils import fetchPly
from r2_gaussian.utils.system_utils import searchForMaxIteration
from r2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks

def initialize_random_gaussians(gaussians: GaussianModel, 
                               offOrigin, 
                               sVoxel, 
                               n_points=50000, 
                               density_value=0.1):
    """Initialize Gaussians with random points within the volume bounds.
    
    Args:
        gaussians: The GaussianModel to initialize
        scanner_cfg: Scanner configuration containing volume information
        n_points: Number of random points to generate
        density_value: Initial density value for all points
    """
    # Get volume bounds from scanner config
    center = np.array(offOrigin)
    size = np.array(sVoxel)
    
    # Generate random points within the volume bounds
    min_bound = center - size/2
    max_bound = center + size/2
    
    # Random positions
    xyz = np.random.uniform(
        low=min_bound, 
        high=max_bound, 
        size=(n_points, 3)
    )
    
    # Initial densities - can now be negative if allowed
    densities = np.random.uniform(
        low=-density_value,
        high=density_value,
        size=(n_points, 1)
    )
    
    # Initialize the Gaussian model
    gaussians.create_from_pcd(xyz, densities, spatial_lr_scale=1.0)
    
    print(f"Initialized {n_points} random Gaussians within volume bounds")
    return gaussians

def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, loaded_iter=None, random_init=False):
    print(args)
    if loaded_iter:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(
                osp.join(args.model_path, "point_cloud")
            )
        ply_path = os.path.join(
            args.model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.pickle",  # Pickle rather than ply
        )

        print("Loading trained model at iteration {}".format(loaded_iter))
    else:
        random = False
        if random:
            initialize_random_gaussians(
                gaussians, 
                args['offOrigin'],
                args['sVoxel'],
                # n_points=args['n_random_points'], 
                # density_value=args['random_density']
            )
        else:
            if args.ply_path == "":
                if osp.exists(osp.join(args.source_path, "meta_data.json")):
                    ply_path = osp.join(
                        args.source_path, "init_" + osp.basename(args.source_path) + ".npy"
                    )
                elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
                    ply_path = osp.join(
                        osp.dirname(args.source_path),
                        "init_" + osp.basename(args.source_path).split(".")[0] + ".npy",
                    )
                else:
                    raise ValueError("Could not recognize scene type!")
            else:
                ply_path = args.ply_path
            # ply_path = '/root/r2_gaussian/data/synthetic_dataset/parallel_ntrain_75_angle_360/cryo_128_0_1_parallel/init_cryo_128_0_1_parallel.npy'
            assert osp.exists(
                ply_path
            ), f"Cannot find {ply_path} for initialization. Please specify a valid ply_path or generate point cloud with initialize_pcd.py."

            print(f"Initialize Gaussians with {osp.basename(ply_path)}")
            ply_type = ply_path.split(".")[-1]
            if ply_type == "npy":
                point_cloud = np.load(ply_path)
                xyz = point_cloud[:, :3]
                density = point_cloud[:, 3:4]
            elif ply_type == ".ply":
                point_cloud = fetchPly(ply_path)
                xyz = np.asarray(point_cloud.points)
                density = np.asarray(point_cloud.colors[:, :1])

            gaussians.create_from_pcd(xyz, density, 1.0)

    return loaded_iter
