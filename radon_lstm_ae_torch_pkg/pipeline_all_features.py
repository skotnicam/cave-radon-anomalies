from radon_lstm_ae_torch_pkg.data import prepare_data
from radon_lstm_ae_torch_pkg.pipeline_common import setup_pipeline
from radon_lstm_ae_torch_pkg.reporting import print_summary, save_outputs_and_plots
from radon_lstm_ae_torch_pkg.training import train_model


def run_pipeline(args):
    device = setup_pipeline(args)
    prepared_data = prepare_data(args)
    # Train the original autoencoder that reconstructs every feature in the window.
    training_artifacts = train_model(args, prepared_data, device)
    report_artifacts = save_outputs_and_plots(args, prepared_data, training_artifacts, device)
    print_summary(args, device, prepared_data, report_artifacts)
