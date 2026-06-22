from pandas import read_csv
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from matplotlib.pyplot import scatter, plot, xlabel, ylabel, title, show
from math import sqrt
#from numpy import values


def read_data():
    filepath = Path(__file__).parent.parent / 'iris.data.csv'
    data = read_csv(filepath)

    X = data.iloc[:, :-1].to_numpy().tolist()

    y = data.iloc[:, -1]
    mapping = {v: i for i, v in enumerate(sorted(y.unique()), start=1)}
    y = y.map(mapping).to_numpy().reshape(-1, 1).tolist()

    return X, y


def transpose(X):
    col = len(X)
    row = len(X[0])

    X_trans = [[0]*col for _  in range(row)]

    for i in range(row):
        for j in range(col):
            X_trans[i][j] = X[j][i]

    return X_trans

def matrix_multiply(x,y):
    row_x = len(x)
    col_x = len(x[0])
    row_y = len(y)
    col_y = len(y[0])  
    if col_x != row_y:
        return "Error: row, column mismatch"
    row_col = col_x

    xy = [[0]*col_y for _ in range(row_x)]

    for i in range(row_x):
        for j in range(row_col):
            for k in range(col_y):
                xy[i][k] += x[i][j]*y[j][k]
    
    return xy


def lu_decompose(A):
    n = len(A)
    upper_mat = [[0.0 for _ in range(n)] for _ in range(n)]
    lower_mat = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for pivot in range(n):
        lower_mat[pivot][pivot] = 1.0

        for col in range(pivot, n):
            sum_u = 0
            for k in range(pivot):
                sum_u += lower_mat[pivot][k]*upper_mat[k][col]
            upper_mat[pivot][col] = A[pivot][col]-sum_u
        
        for col in range(pivot, n):
            sum_l = 0
            for k in range(pivot):
                sum_l += lower_mat[col][k]*upper_mat[k][pivot]
            lower_mat[col][pivot] = (A[col][pivot]-sum_l)/upper_mat[pivot][pivot]
            
    return lower_mat, upper_mat
    

def lower_liner_solver(L, B):
    n = len(B)
    Y = [[0.0] for _ in range(n)]
    for i in range(n):
        sum_y = 0
        for j in range(i):
            sum_y += Y[j][0]*L[i][j]
        Y[i][0] = B[i][0] - sum_y

    return Y

def upper_linear_solver(U, Y):
    n = len(Y)
    X = [[0.0] for _ in range(n)]
    for i in range(n-1,-1,-1):
        sum_x=0
        for j in range(n-1,i,-1):
            sum_x += X[j][0]*U[i][j]
        X[i][0] = (Y[i][0]-sum_x)/U[i][i]

    return X


def linear_solver(A, B):
    L, U = lu_decompose(A)
    Y = lower_liner_solver(L, B)
    X = upper_linear_solver(U, Y)
    return X
    


def linear_regression(X, y):
    X_trans = transpose(X)

    XTX = matrix_multiply(X_trans, X)

    XTy = matrix_multiply(X_trans, y)

    beta = linear_solver(XTX, XTy)
    
    print("Linear Equation y = " + " + ".join(f"{b[0]:.2f}*x{i+1}" for i,b in enumerate(beta)).replace("+ -","- "))

    return beta
    
def y_plot(y_test, y_pred):
    scatter(y_test, y_pred, c='blue')
    plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    xlabel('Actual y')
    ylabel('Predicted y')
    title('Actual vs predicted')
    show()

def perfomance_scores(y_test, y_pred):
    y_plot(y_test, y_pred)
    n = len(y_pred)

    mean_absolute_error = sum(abs(y_test[i][0]-y_pred[i][0]) for i in range(n))/n
    print(f"Mean Absolute Error: {mean_absolute_error:.2f}")

    mean_square_error = sum((y_test[i][0]-y_pred[i][0])**2 for i in range(n))/n
    print(f"Mean Sqaure Error: {mean_square_error:.2f}")

    root_mean_square_error = sqrt(mean_square_error)
    print(f"Root mean Sqaure Error: {root_mean_square_error:.2f}")

    mean = sum(y_test[i][0] for i in range(n))/n
    residual_sum_sqaures = n*mean_square_error
    total_sum_squares = sum((y_test[i][0]-mean)**2 for i in range(n))
    r_squared = 1 - (residual_sum_sqaures/total_sum_squares)
    print(f"R Squared Value: {r_squared:.2f}")

    return {
        'mae': mean_absolute_error,
        'mse': mean_square_error,
        'rmse': root_mean_square_error,
        'r2': r_squared
    }



def kFoldValidation():
    X, y = read_data()
    kf = KFold(n_splits=3, shuffle=True, random_state=2)
    
    all_metrics = {'mae': [], 'mse': [], 'rmse': [], 'r2': []}
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\n=== Fold {fold} ===")
        X_train, X_test = X[train_index].tolist(), X[test_index].tolist()
        y_train, y_test = y[train_index].tolist(), y[test_index].tolist()

        beta = linear_regression(X_train, y_train)
        y_pred = matrix_multiply(X_test, beta)
        
        metrics = perfomance_scores(y_test, y_pred)
        
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    print("\n" + "="*50)
    print("COMBINED RESULTS (All Folds)")
    print("="*50)
    n_folds = len(all_metrics['mae'])
    for metric_name, values in all_metrics.items():
        mean_val = sum(values) / n_folds
        std_val = sqrt(sum((v - mean_val)**2 for v in values) / n_folds)
        print(f"{metric_name.upper():>5}: {mean_val:.4f} ± {std_val:.4f}")
    print("="*50)



def main():
    kFoldValidation()


if __name__ == "__main__":
    main()
