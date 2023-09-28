import click
from utils import get_split,  load_data
from methods import run_methods


@click.command()
@click.option('--patho', help='pathology')
@click.option('--method', help='which method of {lime, shap} should be used, if None, both are used')
@click.option('--model', multiple=True, help='which model of {rf, xgb, lr, dn, gp} \
should be used, if None, all are used')
@click.option('--reduced', default=False, is_flag=True, help='just relevant for lvh, flags whether to use reduced or full dataset')
@click.option('--output_dir', default="./output")
def run_binary_feature_importance_experiments(patho, method, model, reduced, output_dir):
    pathos = ['avblock', 'lbbb', 'rbbb'] if patho is None else [patho]
    for patho in pathos:
        df, tag = load_data(patho, reduced)

        X_train, X_test, y_train, y_test, features = get_split(df, patho)

        models = ['rf', 'xgb', 'lr', 'dn', 'gp'] if len(model) == 0 else model

        run_methods(output_dir, patho, method, models, X_train,
                    X_test, y_train, y_test, features, tag=tag)


if __name__ == '__main__':
    run_binary_feature_importance_experiments()
