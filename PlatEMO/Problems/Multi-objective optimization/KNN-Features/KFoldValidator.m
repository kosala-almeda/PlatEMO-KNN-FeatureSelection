% K-Fold Validation
% Inputs:
%   k: number of folds
%   data: data to be split into folds
%   labels: labels for data
%   classifier: classifier to be validated
%   randomize: whether to randomize data before splitting into folds
% Outputs:
%   accuracy: average accuracy of classifier
%   folds: cell array of folds, each containing a data matrix and a labels
%
% References:
%    - https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
%    - https://www.mathworks.com/help/stats/crossval.html
%    - https://github.com/amoudgl/kNN-classifier



classdef KFoldValidator
    properties
        k
        classifier
        randomize
    end
    
    methods
        function obj = KFoldValidator(k, classifier, randomize)
            if nargin < 3
                randomize = true;
            end
            obj.k = k;
            obj.classifier = classifier;
            obj.randomize = randomize;
        end
        
        function [accuracy] = validate(obj, data, labels)
            % Split data into k folds
            folds = obj.splitData(data, labels);
            
            % Run k-fold validation
            accuracy = 0;
            for i = 1:obj.k
                % Get training and test data
                trainingData = [];
                trainingLabels = [];
                for j = 1:obj.k
                    if j ~= i
                        trainingData = [trainingData; folds{j}.data];
                        trainingLabels = [trainingLabels; folds{j}.labels];
                    end
                end
                testData = folds{i}.data;
                testLabels = folds{i}.labels;
                
                % Train classifier
                obj.classifier = obj.classifier.fit(trainingData, trainingLabels);
                
                % Test classifier
                predictedLabels = obj.classifier.predict(testData);
                
                % Calculate accuracy
                accuracy = accuracy + sum(predictedLabels == testLabels) / length(testLabels);
            end
            accuracy = accuracy / obj.k;
        end
        
        function folds = splitData(obj, data, labels)
            % Shuffle data
            if obj.randomize
                shuffledIndices = randperm(length(data));
                shuffledData = data(shuffledIndices, :);
                shuffledLabels = labels(shuffledIndices);
            else
                shuffledData = data;
                shuffledLabels = labels;
            end
            
            % Split data into k folds
            folds = cell(obj.k, 1);
            for i = 1:obj.k
                folds{i}.data = shuffledData(i:obj.k:end, :);
                folds{i}.labels = shuffledLabels(i:obj.k:end);
            end
        end
    end
end