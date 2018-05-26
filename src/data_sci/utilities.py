from .imports import *


def display_all(df):
    """
    Display all the rows and columns of the dataframe
    :param df: Dataframe
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


def display_some(df,rows,cols):
    """
    Display a given number of rows and columns of the dataframe

    :param df: Dataframe
    :param rows: Number of rows
    :param cols: Number of columns
    """
    with pd.option_context("display.max_rows", rows, "display.max_columns", cols):
        display(df)


def split_train_val(df, trn_amount, keep_order=True):
    """
    Split a dataframe in two

    :param df: Dataframe
    :param trn_amount: If a value between 0 and 1, corresponds to the percentage of the df to be used in training, if
        greater than corresponds to the number of rows in the df to be used in training
    :param keep_order: If True (default) will split the dataframe so that the training set is rows [:n] and the
        validation set is rows [n:]
    :return df_train: Portion of the dataframe used for training
    :retrun df_valid: Portion of the dataframe used for validation
    """

    # try:
    #     assert 0 <= trn_prcnt <= 1
    # except:
    #     print("trn_prcnt must be between 0 and 1")
    #     raise
    if 0 <= trn_amount <= 1:
        n_trn = int(len(df)*trn_amount)
    else:
        n_trn = trn_amount
    if not keep_order:
        df = df.sample(frac=1)
    df_train = df[:n_trn]
    df_valid = df[n_trn:]
    return [df_train, df_valid]


def custom_RFscore(m, train_set, valid_set, score_func='rmse', use_oob=False):
    """
    Returns a score based on custom function for the sklearn random forest regressor

    :param m: RandomForestRegressor object output of sklearn
    :param train_set: training set dataframe without the output column
    :param valid_set: Output column of training set as dataframe
    :param use_oob: (optional) Whether or not to use the out of bag score function in sklearn's RF.  It is the average
        error for each training observation (x_i,y_i) calculated using predictions from the trees that do not contain
        that training observation in their respective bootstrap sample.
    :param score_func: what function to use for calculating the score.  Options are mse, rmse (Default)
    :return score: Fit score using score function
    """
    train_pred = m.predict(train_set)
    if score_func == 'rmse':
        score = np.sqrt(((train_pred-valid_set)**2).mean())
    elif score_func == 'mse':
        score = ((train_pred - valid_set) ** 2).mean()
    if use_oob and hasattr(m, 'oob_score_'):
        return score, m.oob_score_
    return score
