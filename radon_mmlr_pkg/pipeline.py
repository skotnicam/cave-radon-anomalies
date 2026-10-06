from radon_mmlr_pkg.mmlr import fit_mmlr
from radon_mmlr_pkg.pipeline_common import setup_pipeline
from radon_mmlr_pkg.reporting import print_summary, save_outputs_and_plots


def run_pipeline(args):
    device = setup_pipeline(args)
    artifacts = fit_mmlr(args)
    report_artifacts = save_outputs_and_plots(args, artifacts)
    print_summary(args, device, artifacts, report_artifacts)
