from radon_lstm_ae_torch_pkg.data import prepare_data
from radon_lstm_ae_torch_pkg.pipeline_common import setup_pipeline
from radon_lstm_ae_torch_pkg.reporting import print_summary, save_outputs_and_plots
from radon_lstm_ae_torch_pkg.training import train_conv_model


def run_pipeline(args):
    device = setup_pipeline(args)
    # Convolutional baseline: reconstruct the full multivariate window with Conv1D blocks.
    prepared_data = prepare_data(args)
    training_artifacts = train_conv_model(args, prepared_data, device)
    report_artifacts = save_outputs_and_plots(
        args,
        prepared_data,
        training_artifacts,
        device,
        method_key="conv_reconstruct_all_features",
        method_name="Conv reconstruction",
        extra_model_payload={"architecture": "conv1d"},
    )
    print_summary(args, device, prepared_data, report_artifacts)
