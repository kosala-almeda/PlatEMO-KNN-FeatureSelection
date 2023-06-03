% KNN Classifier
% Inputs:
%    k: number of neighbors
%    X: matrix with training data
%    y: vector with labels of training data
%    d: distance metric (default: manhattan)
% Outputs:
%    y_pred: vector with predicted labels
%
% References:
%    - https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
%    - https://www.mathworks.com/help/stats/classification-using-nearest-neighbors.html
%    - https://github.com/amoudgl/kNN-classifier

classdef KNNClassifier
    properties
        k
        X
        y
        d = 'manhattan'
    end
    
    methods
        function obj = KNNClassifier(k, d)
            if nargin > 1
                obj.d = d; % set distance metric
            end
            obj.k = k;
        end
        
        function obj = fit(obj, X, y)
            obj.X = X;
            obj.y = y;
        end
        
        function y_pred = predict(obj,  X)
            % initialize y_pred
            y_pred = zeros(size(X, 1), 1);
            % loop over given data
            for i = 1:size(X, 1)
                % calculate distances
                if strcmp(obj.d, 'euclidean') % euclidean distance
                    distances = sqrt(sum((obj.X - X(i, :)).^2, 2));
                else % manhattan distance is the default
                    distances = sum(abs(obj.X - X(i, :)), 2);
                end
                % sort distances and get the k nearest neighbors
                [~, indices] = sort(distances);
                k_nearest = obj.y(indices(1:obj.k));
                % get the most frequent label and handle ties
                y_pred(i) = mode(k_nearest);
            end
        end
    end
end