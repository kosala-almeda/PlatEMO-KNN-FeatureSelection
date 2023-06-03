classdef KNNFS < PROBLEM
% <multi> <binary/integer> <expensive/none>
% Feature selection for KNN classification
% Kn      ---     5 --- K value of KNN
% Kf      ---     4 --- K value of K-fold cross validation

methods
    function Setting(obj)
        obj.M = 2;
        obj.D = 1;
        if isempty(obj.Parameter)
            obj.Parameter = {K,D};
        else
            obj.Parameter = obj.Parameter(1:2);
        end
    end
end

end