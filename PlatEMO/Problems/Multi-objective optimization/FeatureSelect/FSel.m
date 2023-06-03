classdef FSel < PROBLEM
% <multi> <binary> <large/none> <expensive/none> <sparse/none>
% The feature selection problem for KNN classification
% dataSetNo --- 1 --- Data set number (1-Hillvally, 2-musk, 3-Madelon, 4-movement)

%------------------------------- Reference --------------------------------
% Y. Tian, X. Zhang, C. Wang, and Y. Jin, An evolutionary algorithm for
% large-scale sparse multi-objective optimization problems, IEEE
% Transactions on Evolutionary Computation, 2020, 24(2): 380-393.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
% UPDATED BY 
% P. D. K. M. Almeda and C. L. Hewa Palihakkarage 
% in 2023 for CPSC5207 Laurentian University

% The datasets are taken from the penfwang repository in
%  https://github.com/penfwang/Inf_Sci_MODE
% No.   Name                              Samples Features Classes
% 1     Hillvally                          1212     100       2
% 2     Musk1                              2031     166       2
% 3     Madelon                            2600     500       2
% 4     Movement                            360      90      15


    properties(Access = private)
        TrainIn;    % Input featues of training set
        TrainOut;   % Output class of training set
        ValidIn;    % Input featues of validation set
        ValidOut;   % Output class of validation set
        TestIn;     % Input featues of test set
        TestOut;    % Output class of test set
        Category;   % Class labels
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            % Load data
            dataSetNo = obj.ParameterSet(1);
            switch dataSetNo
                case 1
                    disp('Hill_Valley');
                    Data1 = importdata('Hill_Valley_without_noise_Training.data').data;
                    Data2 = importdata('Hill_Valley_without_noise_Testing.data').data;
                    fullDataSet = vertcat(Data1, Data2);

                case 2
                    disp('Musk1');
                    file = importdata('musk.csv');
                    fullDataSet = file.data;

                case 3
                    disp('Madelon');
                    file = importdata('madelon.csv');
                    fullDataSet = file.data;

                case 4
                    disp('Movement_libras');
                    fullDataSet = importdata('movement_libras.data');

            end

            % reduce data set to 1000 rows 
            if size(fullDataSet, 1) > 100
                selectedDataSet = fullDataSet(randperm(size(fullDataSet, 1), 1000), :);
            else
                selectedDataSet = fullDataSet;
            end

            [train, valid, test] = obj.segmentData(selectedDataSet);
            [trInput, trOutput] = obj.normalizeData(train);
            [vaInput, vaOutput] = obj.normalizeData(valid);
            [teInput, teOutput] = obj.normalizeData(test);

            obj.Category    = unique(trOutput(:,end));
            obj.TrainIn     = trInput;
            obj.TrainOut    = trOutput;
            obj.ValidIn     = vaInput;
            obj.ValidOut    = vaOutput;
            obj.TestIn      = teInput;
            obj.TestOut     = teOutput;

            % Parameter setting
            obj.M        = 2; % Number of objectives (Number of selected features, Validation error)
            obj.D        = size(obj.TrainIn,2); % Number of decision variables (Number of features in the dataset)
            obj.encoding = 4 + zeros(1,obj.D); % Encoding design (4: binary encoding)
        end

        % Segmantation (train, validation, test)
        function [train, valid, test] = segmentData(obj,dataSet)

            numRows = size(dataSet, 1);  
            
            indicesTrain = randperm(numRows, round(numRows * 0.7));
            train = dataSet(indicesTrain, :);  
            
            remainingIndices = setdiff(1:numRows, indicesTrain);  
            remainingData = dataSet(remainingIndices, :); 
            
            indicesValid = randperm(size(remainingData, 1), round(numRows * 0.15));  
            valid = remainingData(indicesValid, :);
            
            test = setdiff(remainingData, valid, 'rows'); 

        end

        % Normalization (min-max normalization)
        function [input, output] = normalizeData(obj, Data)
            Fmin = min(Data(:,1:end-1),[],1);
            Fmax = max(Data(:,1:end-1),[],1);
            Data(:,1:end-1) = (Data(:,1:end-1)-repmat(Fmin,size(Data,1),1))./repmat(Fmax-Fmin,size(Data,1),1);
            input = Data(:,1:end-1);
            output = Data(:,end);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            PopDec = logical(PopDec);
            PopObj = zeros(size(PopDec,1),2);
            for i = 1 : size(PopObj,1)
                % Rank the training samples according to their distances to the current solution
                [~,Rank] = sort(pdist2(obj.ValidIn(:,PopDec(i,:)),obj.TrainIn(:,PopDec(i,:))),2);
                % Predict the labels by the majority voting of K(=5) Nearest Neighbors
                %   mode is not used as following strategy is better at tie breaking
                %   it gives priority close neighbors in case of a tie
                [~,Out]  = max(hist(obj.TrainOut(Rank(:,1:5))',obj.Category),[],1); 
                Out      = obj.Category(Out);
                % Using mean over sum to normalize the objective value between 0 and 1
            	PopObj(i,1) = mean(PopDec(i,:));
                % Validation error is the ratio of misclassified samples
                PopObj(i,2) = mean(Out~=obj.ValidOut);
            end
        end
        %% Display a population in the objective space
        function DrawObj(obj,Population)
            % Rescale the objective values
            PopObj = Population.objs;
            PopObj(:,1) = PopObj(:,1)*obj.D;
            PopObj(:,2) = PopObj(:,2)*100;
            Draw(PopObj,{'No. of selected features','Validation error %',[]});
        end
    end
end