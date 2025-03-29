include("building_tree.jl")  # Assuming the new function is in this file
include("utilities.jl")
include("merge.jl")

function main_merge_purity()
    for dataSetName in ["iris", "seeds", "wine"]  
        
        print("=== Dataset ", dataSetName)
        
        # Data preparation (unchanged)
        include("data/" * dataSetName * ".txt")
        reducedX = Matrix{Float64}(X)
        for j in 1:size(X, 2)
            reducedX[:, j] .-= minimum(X[:, j])
            reducedX[:, j] ./= maximum(X[:, j])
        end

        train, test = train_test_indexes(length(Y))
        X_train = reducedX[train,:]
        Y_train = Y[train]
        X_test = reducedX[test,:]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), 
                ", features: ", size(X_train, 2), ")")
        
        # Increased time limit for more reliable results
        time_limit = 300  # 5 minutes instead of 10 seconds

        for D in 2:4
            println("\tD = ", D)
            
            # Univariate case
            println("\t\tUnivarié")
            println("\t\t\t- Iterative heuristic (FhS)")
            testMergePurity(X_train, Y_train, X_test, Y_test, D, classes, 
                          time_limit=time_limit, isMultivariate=false, isExact=false)
            println("\t\t\t- Iterative exact (FeS)")
            testMergePurity(X_train, Y_train, X_test, Y_test, D, classes, 
                          time_limit=time_limit, isMultivariate=false, isExact=true)
            
            # # Multivariate case
            # println("\t\tMultivarié") 
            # testMergePurity(X_train, Y_train, X_test, Y_test, D, classes,
            #               time_limit=time_limit, isMultivariate=true)
        end
    end
end

function testMergePurity(X_train, Y_train, X_test, Y_test, D, classes;
                       time_limit::Int=-1, isMultivariate::Bool=false, alpha::Float64=1.0, isExact::Bool=false)    
    println("\t\t\tGamma\t#Clust\tGap\tTrainErr/TestErr\tTime(s)")
    
    for gamma in 0:0.2:1
        print("\t\t\t", gamma * 100, "%\t")
        
        # Get clusters
        clusters = simpleMerge(X_train, Y_train, gamma)
        print(length(clusters), "\t")
        

        # Build tree with new objective
        T, obj, resolution_time, gap = build_tree_purity(clusters, D, classes, multivariate=isMultivariate,time_limit=time_limit, alpha = alpha, useFhS = !isExact, useFeS = isExact )
        
        # Print results
        print(round(gap, digits=1), "%\t")
        print(prediction_errors(T,X_train,Y_train, classes), "/" , prediction_errors(T,X_test,Y_test, classes), "\t")
        println(round(resolution_time, digits=1), "s")
    end
    println() 
end