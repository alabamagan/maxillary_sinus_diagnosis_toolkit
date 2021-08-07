import pandas as pd
import os
import pathlib
from .console_entry import ConsoleEntry
from ..pipeline.steps import Seg2Diag, data_preparation

__all__ = ['train_seg2diag', 'inference_seg2diag']

def train_seg2diag():
    parser = ConsoleEntry('OLngv')
    parser.add_argument('-s1', '--s1-res', action='store', 
                        help="Label statistics of stage 1 outputs (cropped).")
    parser.add_argument('-s2', '--s2-res', action='store', 
                        help='Label statistics of stage 2 outputs (cropped).')
    parser.add_argument('-gt', '--ground-truth', action='store',
                        help='CSV or Excel file that contains the ground truth.')
    parser.add_argument('--seg2d-dict', action='store', default=None,
                        help='Specify the dict that map lesion name to column name in the ground-truth  file.')
    args = parser.parse_args()

    # Check data availability
    assert os.path.isfile(args.s1_res) and os.path.isfile(args.s2_res), "Inputs are not correctly specified."
    assert os.path.isfile(args.ground_truth), f"Can't open specified ground-truth file at: {args.ground_truth}"

    df = data_preparation(args.s1_res, args.s2_res, args.ground_truth)
    parser.logger.info("Trying to fit data.")

    model = Seg2Diag()
    if args.seg2d_dict is not None:
        assert isinstance(args.seg2d_dict, dict), "Argument seg2d-dict must be dictionary."
        model.default_dict = args.seg2d_dict
    model.fit(df)

    parser.logger.info(f"Saving fitted model to {args.outfile}")
    model.save(args.outfile)

def inference_seg2diag():
    parser = ConsoleEntry('OLngv')
    parser.add_argument('-s1', '--s1-res', action='store',
                        help="Label statistics of stage 1 outputs (cropped).")
    parser.add_argument('-s2', '--s2-res', action='store',
                        help='Label statistics of stage 2 outputs (cropped).')
    parser.add_argument('-i', '--saved-model', action='store',
                        help='Directory to model.')
    parser.add_argument('-gt', '--ground-truth', action='store', default=None,
                        help='CSV or Excel file that contains the ground truth.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the resultant ROC curve. Require specifying ground-truth.')
    parser.add_argument('--seg2d-dict', action='store', default=None,
                        help='Specify the dict that map lesion name to column name in the ground-truth  file.')
    parser.add_argument('--use-training-cutoff', action='store_true',
                        help='If true, use cutoff computed from training data.')
    args = parser.parse_args()

    # Check data availability
    assert os.path.isfile(args.s1_res) and os.path.isfile(args.s2_res), "Inputs are not correctly specified."

    df = data_preparation(args.s1_res, args.s2_res)
    parser.logger.info("Trying to fit data.")
    model = Seg2Diag()
    model.load(args.saved_model)
    predict = model.predict(df)

    # If ground-truth data is provided.
    if not args.ground_truth is None:
        assert os.path.isfile(args.ground_truth), f"Can't open specified ground-truth file at: {args.ground_truth}"

        df_gt = pd.read_csv(args.ground_truth)
        df_gt = df_gt.astype({'Patient ID': str})
        df_gt.set_index(['Patient ID', 'Right/Left'], drop=True, inplace=True)
        df_gt = df_gt.loc[df.index]

        # Use cut off deduced from ground-truth rather than the trained model
        if not args.use_training_cutoff:
            model.compute_cutoff(df, df_gt)

        if args.plot:
            model.plot_model_results(df, df_gt)


    # Generate prediction report
    out_cols = []
    df_out = pd.DataFrame(df)
    df_cutoff = pd.DataFrame()

    out = {key: predict[key] >= model.cutoffs[key] for key in model.default_dict}
    for key in predict:
        df_out[f'SVR Score - {key}'] = predict[key]
        df_out[f'SVR Prediction - {key}'] = out[key]
        out_cols.extend([f'SVR Score - {key}', f'SVR Prediction - {key}'])
        df_cutoff = df_cutoff.append(pd.Series(name=key, data=[model.cutoffs[key]]))

    if args.outfile is not None:
        output_path = args.outfile
        output_path = output_path + '.xlsx' if not output_path.endswith('.xlsx') else output_path

        with pd.ExcelWriter(output_path) as writer:
            df_out[out_cols].to_excel(writer, sheet_name='SVR result')
            df_cutoff.to_excel(writer, sheet_name='Youden Cutoffs')
            writer.save()

    return df_out[out_cols], df_cutoff

