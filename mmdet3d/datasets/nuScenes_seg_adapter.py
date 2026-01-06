# Adapter to ensure NuScenesSegDataset from projects/TPVFormer is imported
# so that the class is registered in DATASETS when configs reference it.
try:
    # Try to import the NuScenesSegDataset implemented in projects/TPVFormer
    from projects.TPVFormer.tpvformer.nuscenes_dataset import NuScenesSegDataset  # noqa: F401
except Exception as e:  # pragma: no cover - helpful import-time message
    raise ImportError(
        'Failed to import NuScenesSegDataset from projects/TPVFormer.\n'
        'If you want to use `NuScenesSegDataset` in configs, make sure the '
        'projects/TPVFormer package is available/importable (e.g., run from '
        'repo root or install the project). Original error: ' + str(e))
