from radon_emd_svr_pkg.emd_svr import fit_emd_svr
from radon_emd_svr_pkg.pipeline_common import setup_pipeline
from radon_emd_svr_pkg.reporting import print_summary, save_outputs_and_plots


def run_pipeline(args):
    device = setup_pipeline(args)
    artifacts = fit_emd_svr(args)
    report_artifacts = save_outputs_and_plots(args, artifacts)
    print_summary(args, device, artifacts, report_artifacts)
