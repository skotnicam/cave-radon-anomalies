from radon_lstm_ae_torch_pkg.data import (
    build_model_feature_cols,
    prepare_reconstruction_supervised_data,
    radon_model_col,
    resolve_base_feature_cols,
)
from radon_lstm_ae_torch_pkg.pipeline_common import setup_pipeline
from radon_lstm_ae_torch_pkg.reporting import print_summary, save_supervised_outputs_and_plots
from radon_lstm_ae_torch_pkg.training import make_transformer_autoencoder, train_supervised_model


def run_pipeline(args):
    device = setup_pipeline(args)
    resolve_base_feature_cols(args)
    input_feature_cols = build_model_feature_cols(args)
    target_feature_cols = [radon_model_col(args)]
    # Transformer alternative using delayed outdoor temperature in the input window.
    prepared_data = prepare_reconstruction_supervised_data(
        args,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
    )
    model = make_transformer_autoencoder(
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
        method_key="transformer_reconstruct_radon_with_delayed_outdoor_temperature",
        method_name="Transformer radon reconstruction (delayed outdoor temperature)",
        extra_model_payload={
            "temp_outdoor_lag_hours": args.temp_outdoor_lag_hours,
            "replace_temp_outdoor_with_lag": bool(args.replace_temp_outdoor_with_lag),
        },
    )
    print_summary(args, device, prepared_data, report_artifacts)
