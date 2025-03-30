using Distances

"""
Représente une distance entre deux données identifiées par leur id
"""
mutable struct Distance
    distance::Float64
    ids::Vector{Int}
    metric_type::Symbol  # Nouveau champ pour stocker le type de métrique

    function Distance()
        return new()
    end
end 

"""
Constructeur d'une distance avec choix de métrique

Entrées :
- id1 : id de la première donnée
- id2 : id de la seconde donnée
- x : caractéristique des données d'entraînement
- metric : type de distance (:euclidean, :manhattan, :chebyshev, :cosine) 
            (euclidean par défaut)
"""
function Distance(id1::Int, id2::Int, x::Matrix{Float64}; metric::Symbol=:euclidean)
    d = Distance()
    d.ids = [id1, id2]
    d.metric_type = metric
    
    # Calcul de la distance selon le type spécifié
    if metric == :euclidean
        d.distance = euclidean(x[id1, :], x[id2, :])
    elseif metric == :manhattan
        d.distance = cityblock(x[id1, :], x[id2, :])
    elseif metric == :chebyshev
        d.distance = chebyshev(x[id1, :], x[id2, :])
    elseif metric == :cosine
        d.distance = 1 - cosine_dist(x[id1, :], x[id2, :])
    else
        error("Métrique non supportée. Choisissez parmi :euclidean, :manhattan, :chebyshev, :cosine")
    end
    
    return d
end