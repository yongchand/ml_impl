from nn import *

iterations = 500
alpha = 0.1


def main():
    X_train_train, y_train_train, X_train_val, y_train_val = import_train_file('train.csv')
    parameter = gradient_descent(X_train_train, y_train_train, 1000, [784, 10, 10], 0.1)
    print("final paramter: ", parameter)
    A_final, caches = forward_full(X_train_val, parameter)
    cost = loss(A_final, y_train_val)
    accuracy = get_accuracy(get_predictions(A_final), y_train_val)
    print("final_cost_and_accuracy: ", cost, accuracy)

if __name__ == "__main__":
    main()