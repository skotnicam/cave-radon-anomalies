import os

from radon_lstm_ae_torch_pkg.data import get_device, set_seed


def setup_pipeline(args):
    # Keep startup concerns in one place so each method pipeline can stay focused on modeling.
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    return device
