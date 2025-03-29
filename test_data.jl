# test functions on datasets
include("main.jl")
include("utilities.jl")




dataset = ["iris", "seeds", "wine"]


for dataSetName in dataset
    print("=== Dataset ", dataSetName)

    # Préparation des données
    include("../data/" * dataSetName * ".txt")

    reducedX = Matrix{Float64}(X)
    for j in 1:size(X, 2)
        reducedX[:, j] .-= minimum(X[:, j])
        reducedX[:, j] ./= maximum(X[:, j])
    end

    train, test = train_test_indexes(length(Y))
    X_train = reducedX[train, :]
    Y_train = Y[train]
    X_test = reducedX[test, :]
    Y_test = Y[test]
    classes = unique(Y)

    println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")

    time_limit = 30

    # Pour chaque profondeur considérée
    for D in 2:4

        println("  D = ", D)

        ## 1 - Univarié (séparation sur une seule variable à la fois)
        # Création de l'arbre
        print("    Univarié...  \t")
        T, obj, resolution_time, gap = build_tree(X_train, Y_train, D,  classes, multivariate = false, time_limit = time_limit)

        # Test de la performance de l'arbre
        print(round(resolution_time, digits = 1), "s\t")
        print("gap ", round(gap, digits = 1), "%\t")
        if T !== nothing
            print("Erreurs train/test ", prediction_errors(T,X_train,Y_train, classes))
            print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        end
        println()

        ## 2 - Multivarié
        print("Multivarié...\t")
        T, obj, resolution_time, gap = build_tree(X_train, Y_train, D, classes, multivariate = true, time_limit = time_limit)
    end
end