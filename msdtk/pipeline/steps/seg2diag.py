from sklearn import svm, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import pandas as pd
import itertools
from joblib import dump, load
from msdtk.utils import norm_df_cols
from typing import List
import numpy as np

__all__ = ['data_preparation', 'Seg2Diag']

def data_preparation(s1_res, s2_res, gt=None):
    r"""
    Prepare datasheet used for training/inference.

    The results from first stage should contains the columns: ['Volume_1', 'Perimeter_1'], which
    can be obtained from calling `generate_label_statistics` function or by `msdtk-label_statistic`
    command. Label 1 in this file corresponds to the air space in the sinus.

    The results from the second stage should contain two labels and their stats, which corresponds
    to mucosal thickening (1) and cyst respectively (2).

    Args:
        s1_res (str):
            Directory to the label statistic csv of stage 1 output.
        s2_res (str):
            Directory to the label statistic csv of stage 2 output.
        gt (str):
            Directory of ground truth regarding thickening and cyst status

    Returns:
        pd.DataFrame

    Examples:
        For `s1_res` and `s2_res`, the table could look like this:

        +------------+-------------+-------------+-------------+-------------+----------+----------+
        | Patient ID | Perimeter_1 | Perimeter_2 | Roundness_1 | Roundness_2 | Volume_1 | Volume_2 |
        +============+=============+=============+=============+=============+==========+==========+
        | 214081_L   | 3698.447739 | 681.7295917 | 0.768914973 | 0.342845281 | 222813   | 3499     |
        +------------+-------------+-------------+-------------+-------------+----------+----------+
        | 214081_R   | 4112.721777 | 146.6063509 | 0.716475727 | 0.388683455 | 235012   | 0        |
        +------------+-------------+-------------+-------------+-------------+----------+----------+
        | 233224_L   | 4046.113631 | 757.6722372 | 0.724199124 | 0.341720473 | 233044   | 965      |
        +------------+-------------+-------------+-------------+-------------+----------+----------+
        | 233224_R   | 3954.48823  | 949.6655136 | 0.714155514 | 0.328319499 | 220505   | 718      |
        +------------+-------------+-------------+-------------+-------------+----------+----------+

        where the integer _1, _2 and _3 are the classes of the segmentation which the metrics were
        computed. (Columns for _3 was hidden for the sake of line width).

        This table can be generated using the command `msdtk-label_statistics`.

        For `gt`, the table should look like this:

        +------------+------------+---------+------+--------------------+-------+
        | Patient ID | Right/Left | Healthy | Cyst | Mucosal Thickening | Group |
        +============+============+=========+======+====================+=======+
        | 60304      |      R     |    1    |   0  |          0         |   0   |
        +------------+------------+---------+------+--------------------+-------+
        | 60304      |      L     |    1    |   0  |          0         |   0   |
        +------------+------------+---------+------+--------------------+-------+
        | 97421      |      R     |    0    |   0  |          1         |   2   |
        +------------+------------+---------+------+--------------------+-------+
        | 97421      |      L     |    0    |   0  |          1         |   2   |
        +------------+------------+---------+------+--------------------+-------+


    """
    def create_mindex(data_col):
        return pd.MultiIndex.from_tuples([o.split('_') for o in data_col], 
                                         names=('Patient ID', 'Right/Left'))

    s1df = pd.read_csv(s1_res, index_col=0) if isinstance(s1_res, str) else s1_res
    s1df.drop(s1df.index[-2:], inplace=True)
    s1df.index = create_mindex(s1df.index)
    # Only retain air space volumes and perimeter
    s1df = s1df[['Volume_1', 'Perimeter_1']]
    s1df.columns = ['Volume_Air', 'Perimeter_Air']

    s2df = pd.read_csv(s2_res, index_col=0) if isinstance(s2_res, str) else s2_res
    s2df.drop(s2df.index[-2:], inplace=True)
    s2df.index = create_mindex(s2df.index)
    s2df = s2df[['Volume_1', 'Volume_2', 'Perimeter_1', 'Perimeter_2', 'Roundness_1', 'Roundness_2']]
    s2df.columns = ['Volume_MT', 'Volume_MRC', 'Perimeter_MT', 'Perimeter_MRC', 'Roundness_MT', 'Roundness_MRC']

    if not gt is None:
        gtdf = pd.read_csv(gt) if isinstance(gt, str) else gt
        gtdf = gtdf.astype({'Patient ID': str})
        gtdf.set_index(['Patient ID', 'Right/Left'], drop=True, inplace=True)

        df = gtdf.join(s1df, how='right').join(s2df)
    else:
        df = s1df.join(s2df)
    return df



class Seg2Diag(object):
    def __init__(self):
        super(Seg2Diag, self).__init__()

        # This default dict maps the keys to the column name of the target ground-truth in `df`
        self.default_dict = {
            'MT': 'Mucosal Thickening',
            'MRC': 'Cyst',
            # 'Healthy': 'Healthy'
        }
        pass

    def fit(self, df: pd.DataFrame, params=None) -> List[Pipeline]:
        r"""
        For training, the input dataframe should contain the MT and MRC status in the first
        four columns, the rest are the features used for training.
        """


        # Drop Ground truth status to get the features
        X = df.drop(df.columns[:4], axis=1)
        mt_y = df['Mucosal Thickening']
        mrc_y = df['Cyst']
        healthy_y = df['Healthy']

        # Compute class weights
        self.models = {}
        for key in self.default_dict:
            _model = Pipeline([('scaler', StandardScaler()), ('svc', svm.SVR())])
            _model.fit(X, df[self.default_dict[key]])
            self.models[key] = _model

        training_prediction_index = self.predict(X)

        # Compute default cut-off with youden index.
        self.compute_cutoff(X, df)
        return self.models

    def predict(self, X):
        return {m: self.models[m].predict(X) for m in self.default_dict}

    def save(self, outfname:str):
        if not outfname.endswith('.msdtks2d'):
            outfname += '.msdtks2d'
        _save_content = {'models': self.models,
                         'cutoffs': self.cutoffs}

        dump(_save_content, outfname)
        #TODO: also need to save/load the cutoffs

    def load(self, infname):
        _loaded_content = load(infname)
        for key in _loaded_content:
            self.__setattr__(key, _loaded_content[key])

    def compute_cutoff(self,
                       X: pd.DataFrame,
                       Y: pd.DataFrame or dict,
                       method: str = 'youden') -> dict:
        r"""
        This method is called by default in `fit` after the models have been fitted.
        For each of the regressed index with respect to MT, MRC and Healthy, a threshold
        on the ROC curve is computed using the specified method.

        Currently, only the Younden's index is implemented.

        In clinical studies, it is more common to deduce the threshold from the testing set
        such that performance metrics like sensitivity and specificity can be calculate. IKn
        this case this method should be called again after fitting.

        Args:
            X (pd.DataFrame):
                Table for prediction. All columns are used as covariates.
            Y (pd.DataFrame):
                Ground truth. Table should contains the ground truth binary status for the
                prediction which are specified in `self.defulat_dict`.
            method (str):
                (NOT IMPLEMENTED).
        """
        self.cutoffs = {}
        # check Y has target
        assert all([self.default_dict[key] in Y for key in self.default_dict]), "Specifiy"


        if method == 'youden':
            for key in self.default_dict:
                model = self.models[key]
                gt_y = Y[self.default_dict[key]]
                # compute ROC
                roc = roc_curve(gt_y, model.predict(X))
                self.cutoffs[key] = Seg2Diag._cutoff_youdens_j(*roc)
            return self.cutoffs
        else:
            raise ArgumentError("Only 'youden' is available now.")

    def plot_model_results(self,
                           X: pd.DataFrame,
                           Y: pd.DataFrame,
                           ax=None ,**kwargs):
        r"""
        Plot the performance of the model on the given dataset, including t
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        for key in self.default_dict:
            prediction = self.models[key].predict(X)
            gt = Y[self.default_dict[key]]
            roc = roc_curve(gt, prediction)
            auc = roc_auc_score(gt, prediction)

            report = classification_report(gt, prediction >= self.cutoffs[key], output_dict=True)
            sens = report['1']['recall']
            specificity = report['0']['recall']
            ax.plot(roc[0], roc[1], label=f"{key} (AUC: {auc:.02f};Sens:{sens:.02f}, Spec: {specificity:.03f})")


        # Plot other stuff
        ax.plot([0, 1], [0, 1], '--', color='Gray', **kwargs)
        ax.set_ylabel('Sensitivity', fontsize=18)
        ax.set_xlabel('1 - Specificity', fontsize=18)
        ax.set_title('ROC curve for detection of MT and MRC', fontsize=18)
        ax.legend(fontsize=18)
        plt.show()

    @staticmethod
    def _cutoff_youdens_j(fpr,tpr,thresholds):
        j_scores = tpr-fpr
        j_ordered = sorted(zip(j_scores,fpr, tpr, thresholds))
        return j_ordered[-1][1]


#
#
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    df = data_preparation('../../../Sinus/98.Output/SinusSeg_B00-v1.0_train/Cropped_CNN/label_statistics.csv',
                           '../../../Sinus/98.Output/SinusSeg_lesion_only_B00-v1.0_train/Cropped_CNN/label_statistics.csv',
                           '/home/lwong/FTP/2.Projects/9.Sinus/01.RAW/CBCT_AI_case information.csv')

    df_test = data_preparation('../../../Sinus/98.Output/SinusSeg_B00-v1.0/Cropped_CNN/label_statistics.csv',
                               '../../../Sinus/98.Output/SinusSeg_lesion_only_B00-v1.0/Cropped_CNN/label_statistics.csv',
                           '/home/lwong/FTP/2.Projects/9.Sinus/01.RAW/CBCT_AI_case information.csv')

    # df.drop(['Volume_Air', 'Perimeter_Air'], axis=1, inplace=True)
    # df_test.drop(['Volume_Air', 'Perimeter_Air'], axis=1, inplace=True)

    model = Seg2Diag()
    model.fit(df)


    X = df_test.drop(df.columns[:4], axis=1)
    mt_y = df_test['Mucosal Thickening']
    mrc_y = df_test['Cyst']
    healthy_y = df_test['Healthy']

    plt.style.use('ggplot')
    model.compute_cutoff(X, df_test)
    Y = model.predict(X)
    model.plot_model_results(X, df_test)
    X['MT'] = Y['MT']
    X['MRC'] = Y['MRC']
    print(X.to_string())
    print(model.cutoffs)
    model.save('./test_save')