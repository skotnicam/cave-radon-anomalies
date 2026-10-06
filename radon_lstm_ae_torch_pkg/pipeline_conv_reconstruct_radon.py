from radon_lstm_ae_torch_pkg.data import (
    build_model_feature_cols,
    prepare_reconstruction_supervised_data,
    radon_model_col,
    resolve_base_feature_cols,
)
from radon_lstm_ae_torch_pkg.pipeline_common import setup_pipeline
from radon_lstm_ae_torch_pkg.reporting import print_summary, save_supervised_outputs_and_plots
from radon_lstm_ae_torch_pkg.training import make_conv_autoencoder, train_supervised_model


def run_pipeline(args):
    device = setup_pipeline(args)
    resolve_base_feature_cols(args)
    input_feature_cols = build_model_feature_cols(args)
    target_feature_cols = [radon_model_col(args)]
    # Convolutional model sees the full window, but learns to reproduce only radon.
    prepared_data = prepare_reconstruction_supervised_data(
        args,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
    )
    model = make_conv_autoencoder(
        args,
        input_size=len(prepared_data.input_feature_cols),
        output_size=len(prepared_data.target_feature_cols),
    )
    training_artifacts = train_supervised_model(args, model=model, prepared_data=prepared_data, device=device)
    report_artifacts = save_supervised_outputs_and_plots(
        args,
        prepared_data,
        training_artifacts,
        device,
        method_key="conv_reconstruct_radon_from_all_features",
        method_name="Conv radon reconstruction",
        extra_model_payload={"architecture": "conv1d"},
    )
    print_summary(args, device, prepared_data, report_artifacts)
