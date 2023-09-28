import pdb
import click
from utils import create_joined_df, get_split, get_stratified_multiclass_table
from methods import run_methods


@click.command()
@click.option('--patho', default='lbbb', help='pathology')
@click.option('--method', help='which method of {lime, shap} should be used, if None, both are used')
@click.option('--model', multiple=True, help='which model of {rf, xgb, lr, dn} \
should be used, if None, all are used')
def run_multiclass_feature_importance_experiments(patho, method, model):
    df= get_stratified_multiclass_table()
    
    X_train, X_test, y_train, y_test, features = get_split(df, patho)

    models = ['rf', 'xgb', 'lr', 'dn'] if len(model) == 0 else model

    run_methods(patho, method, models, X_train,
                X_test, y_train, y_test, features, tag='multiclass_')


if __name__ == '__main__':
    run_multiclass_feature_importance_experiments()
