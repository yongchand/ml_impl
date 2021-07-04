from nn import *

iterations = 1000
alpha = 0.1


def main():
    X_train_train, y_train_train, X_train_val, y_train_val = import_train_file()
    parameter = gradient_descent(X_train_train, y_train_train, iterations, [784, 10, 10], alpha)
    A_final, caches = forward_full(X_train_val, parameter)
    cost = loss(A_final, y_train_val)
    accuracy = get_accuracy(get_predictions(A_final), y_train_val)
    print("final_cost_and_accuracy: ", cost, accuracy)

    X_test = import_test_file()
    A_final_test, caches_test = forward_full(X_test, parameter)
    y_test = get_predictions(A_final_test)
    write(y_test)

if __name__ == "__main__":
    main()