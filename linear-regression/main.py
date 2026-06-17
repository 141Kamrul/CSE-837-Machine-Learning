from pandas import read_csv
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold


def read_data():
    filepath = Path(__file__).parent.parent / 'iris.data.csv'
    data = read_csv(filepath)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
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
                xy[i][k]+= x[i][j]*y[j][k]
    
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
    


def linear_solver(A, B):
    L, U = lu_decompose(A)
    print(L)
    print(U)
    

def linear_regression(X, y):
    X_trans = transpose(X)
    XTX = matrix_multiply(X_trans, X)
    #XTX_inverse = matrix_inversion(XTX)

    XTy = matrix_multiply(X_trans, y)

    beta = linear_solver(XTX, XTy)

    y_pred = matrix_multiply(beta, X)

    return y_pred
    
def perfomance_scores(y_test, y_pred):
    pass

def kFoldValidation():
    X, y = read_data()
    kf = KFold(n_splits=3, shuffle=True, random_state=2)
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_pred = linear_regression(X_train, y_train)

        perfomance_scores(y_test, y_pred)



def main():
    #kFoldValidation()
    '''x = [[1,2,3], [6,7,8], [4,5,6]]
    xt=transpose(x)
    print(xt)'''
    x = [[1,1,1], [3,1,-3], [1,-2,-5]]
    y = [1,5,10]
    '''xy = matrix_multiply(x,y)
    print(xy)'''
    linear_solver(x,y)
    print("Hello from linear-regression!")


if __name__ == "__main__":
    main()
