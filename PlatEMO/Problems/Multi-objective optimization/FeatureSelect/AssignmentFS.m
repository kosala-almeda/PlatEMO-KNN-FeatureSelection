classdef AssignmentFS < PROBLEM
% <multi> <binary> <large/none> <expensive/none> <sparse/none>
% The feature selection problem for KNN classification
% dataSetNo --- 4 --- Data set number (1-WBCD, 2-Sonar, 3-Movement, 4-Hillvally, 5-Musk1, 6-Multiple(pix), 7-Arrhythmia, 8-Madelon)
% Kf        --- 4 --- Number of folds for KFold cross validation
% Kn        --- 5 --- Number of nearest neighbors for KNN classification
% testSize  --- 0.3 --- Portion of data used for final testing

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
%
% The information of datasets
% Index     Dataset        # Features   # Classes   # Instances     Feature Type    Area
%     1     WBCD                   30           2           569     Real            Medicine
%     2     Sonar                  60           2           208     Real            Physics
%     3     Movement               90          15           360     Real            Medicine
%     4     Hillvally             100           2          1212     Real            Graph
%     5     Musk1                 166           2          2031     Integer         Physics
%     6     Multiple (pix)        240          10          2000     Mixed           Digit Recognition
%     7     Arrhythmia            279          16           452     Mixed           Medicine
%     8     Madelon               500           2          2600     Real            Artificial Dataset

    properties(Access = private)
        TrainIn;    % Input featues of training set
        TrainOut;   % Output class of training set
        TestIn;     % Input featues of test set
        TestOut;    % Output class of test set
        Category;   % Class labels
        Kf;         % Number of folds for KFold cross validation
        Kn;         % Number of nearest neighbors for KNN classification
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            % Parameter setting
            [dataSetNo, obj.Kf, obj.Kn, testSize] = obj.ParameterSet(4, 4, 5, 0.3);

            % Load data
            [inputs,output] = loadDataSet(dataSetNo);

            % shuffle data
            randIdx = randperm(size(inputs, 1));
            inputs = inputs(randIdx, :);
            output = output(randIdx, :);

            % reduce data set to 1000 rows 
            if size(inputs, 1) > 1000
                inputs = inputs(1:1000, :);
                output = output(1:1000, :);
            end

            inputs = normalizeData(inputs);
            obj.Category = unique(output);

            % Split data into training and test sets
            obj.TestIn     = inputs(1:ceil(end*testSize),:);
            obj.TestOut    = output(1:ceil(end*testSize),:);
            obj.TrainIn     = inputs(ceil(end*testSize)+1:end,:);
            obj.TrainOut    = output(ceil(end*testSize)+1:end,:);

            % Parameter setting
            obj.M        = 2; % Number of objectives (Number of selected features, Validation error)
            obj.D        = size(obj.TrainIn,2); % Number of decision variables (Number of features in the dataset)
            obj.encoding = 4 + zeros(1,obj.D); % Encoding design (4: binary encoding)

            display(['# features: ', num2str(obj.D), '   # classes: ', num2str(length(obj.Category)), ...
                '   # train: ', num2str(size(obj.TrainIn,1)), '   # test: ', num2str(size(obj.TestIn,1))])

            % Classification error using all features
            result = CalObj(obj,ones(1,obj.D));
            display(['Classification error using all features: ', num2str(result(2))])
            disp('===============================================')
        end

        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            PopDec = logical(PopDec);
            PopObj = zeros(size(PopDec,1),2);
            for i = 1 : size(PopObj,1)

                % KFold cross validation using Training set
                foldSize = ceil(size(obj.TrainIn,1)/obj.Kf);
                % split to Kf folds, for each fold, train and test
                for j = 1:obj.Kf
                    % get training and test data
                    if j == 1
                        trainIn = obj.TrainIn(foldSize+1:end, :);
                        trainOut = obj.TrainOut(foldSize+1:end, :);
                        validIn = obj.TrainIn(1:foldSize, :);
                        validOut = obj.TrainOut(1:foldSize, :);
                    elseif j == obj.Kf
                        trainIn = obj.TrainIn(1:(j-1)*foldSize, :);
                        trainOut = obj.TrainOut(1:(j-1)*foldSize, :);
                        validIn = obj.TrainIn((j-1)*foldSize+1:end, :);
                        validOut = obj.TrainOut((j-1)*foldSize+1:end, :);
                    else
                        trainIn = obj.TrainIn([1:(j-1)*foldSize, j*foldSize+1:end], :);
                        trainOut = obj.TrainOut([1:(j-1)*foldSize, j*foldSize+1:end], :);
                        validIn = obj.TrainIn((j-1)*foldSize+1:j*foldSize, :);
                        validOut = obj.TrainOut((j-1)*foldSize+1:j*foldSize, :);
                    end
                    

                    % Rank the training samples according to their distances to the current solution
                    [~,Rank] = sort(pdist2(validIn(:,PopDec(i,:)),trainIn(:,PopDec(i,:))),2);
                    % Predict the labels by the majority voting of K Nearest Neighbors
                    %   mode is not used as following gives priority close neighbors in case of a tie
                    [~,Out]  = max(hist(trainOut(Rank(:,1:obj.Kn))',obj.Category),[],1);
                    Out      = obj.Category(Out);
                    % Using mean over sum to normalize the objective value between 0 and 1
                    PopObj(i,1) = mean(PopDec(i,:));
                    % Validation error is the ratio of misclassified samples
                    PopObj(i,2) = mean(Out~=validOut);
                end
            end
        end

        %% Display a population in the objective space
        function DrawObj(obj,Population)
            % Rescale the objective values
            PopObj = Population.objs.*[obj.D,1];
            ParetoObj = Population.best.objs.*[obj.D,1];
            % Get the minumum validation error
            [~,bi] = min(ParetoObj(:,2));
            ax = Draw(PopObj,'o','Markeredgecolor',[0.2 0.2 1],{'No. of selected features','Validation error',[]});
            plot(ax,ParetoObj(:,1),ParetoObj(:,2),'o','Markerfacecolor',[0.5 0.6 1],'Markeredgecolor',[0 0 0.2]);
            plot(ax,ParetoObj(bi,1),ParetoObj(bi,2),'k.');
            ax.XLim = [0,max(PopObj(:,1))+1];
            ax.YLim = [0,max(PopObj(:,2))+0.01];
            legend(ax,'Other Solutions','Pareto front', 'MCE', 'Location', 'SouthWest');
        end
        %% Display a population with post optimization test results
        function DrawTest(obj,Population)
            try
                PopObj = Population.adds;
            catch
                obj.PostOptimization(Population);
                PopObj = Population.adds;
            end
            OrigParetoObj = Population.best.adds;
            OrigParetoObj = OrigParetoObj.*[obj.D,1];
            pi = NDSort(PopObj,1) == 1;
            ParetoObj = PopObj(pi,:).*[obj.D,1];
            [~,bi] = min(ParetoObj(:,2));
            PopObj = PopObj.*[obj.D,1];
            ax = Draw(PopObj,'o','Markeredgecolor',[1 0.5 0.5],{'No. of selected features','Test error',[]});
            plot(ax,OrigParetoObj(:,1),OrigParetoObj(:,2),'o','Markerfacecolor',[1 0.9 0.9],'Markeredgecolor',[0.8 0 0]);
            plot(ax,ParetoObj(:,1),ParetoObj(:,2),'o','Markerfacecolor',[1 0.5 0.5],'Markeredgecolor',[0.2 0 0]);
            plot(ax,ParetoObj(bi,1),ParetoObj(bi,2),'k.');
            ax.XLim = [0,max(PopObj(:,1))+1];
            ax.YLim = [0,max(PopObj(:,2))+0.01];
            legend(ax,'Other Solutions', 'Original Pareto','Non dominated', 'MCE', 'Location', 'SouthWest');
        end
        %% Display a population in the decision space
        function DrawDec(obj,Population)
            ax = Draw(logical(Population.decs));
            PopObjs = Population.objs;
            [~,bi] = min(PopObjs(:,2));
            pi = NDSort(PopObjs,1) == 1;

            PopDecs = Population.decs;
            C = zeros(size(PopDecs)) + 0.8;
            C(pi,:) = zeros+0.5;
            C(bi,:) = zeros;
            C(~PopDecs) = 1;
            surf(ax,ones(size(PopDecs')),repmat(C',1,1,3),'EdgeColor','none');
            legend(ax,'off');
        end
        %% Calculate the error for test set
        function PostOptimization(obj,Population)
            mce = ones(2);
            mcesolution = [];
            for i = 1 : length(Population)
                PopDec = logical(Population(i).dec);
                % Rank the training samples according to their distances to the current solution
                [~,Rank] = sort(pdist2(obj.TestIn(:,PopDec),obj.TrainIn(:,PopDec)),2);
                % Predict the labels by the majority voting of K(=5) Nearest Neighbors
                %   mode is not used as following strategy is better at tie breaking
                %   it gives priority close neighbors in case of a tie
                [~,Out]  = max(hist(obj.TrainOut(Rank(:,1:5))',obj.Category),[],1); 
                Out      = obj.Category(Out);
                % Using mean over sum to normalize the objective value between 0 and 1
            	PopTest(1, 1) = mean(PopDec);
                % Validation error is the ratio of misclassified samples
                PopTest(1, 2) = mean(Out~=obj.TestOut);
                % Store in the population
                Population(i).add = PopTest;
                % Update the minumum classification error
                if PopTest(1, 2) < mce(1)
                    mce(2, 1) = PopTest(1, 2);
                    mce(2, 2) = Population(i).obj(1, 2);
                    mcesolution(2,:) = PopDec;
                end
                % Update the minumum classification error for training data
                if Population(i).obj(1, 2) < mce(2)
                    mce(1, 1) = Population(i).obj(1, 2);
                    mce(1, 2) = PopTest(1, 2);
                    mcesolution(1,:) = PopDec;
                end
            end
            disp('For training data:');
            display(['MCE: ',num2str(mce(1)), '  (test:', num2str(mce(1,2)), ') ,   No of Features: ',num2str(sum(mcesolution(1,:)))]);
            display(['Feature set: ',num2str(find(mcesolution(1,:)))]);
            display(['HV: ', num2str(obj.CalMetric('HV',Population))])
            disp('For test data:');
            display(['MCE: ',num2str(mce(2)), '  (train:', num2str(mce(2,2)), ') ,   No of Features: ',num2str(sum(mcesolution(2,:)))]);
            display(['Feature set: ',num2str(find(mcesolution(2,:)))]);
            disp('===============================================')
        end
    end
end

% Get the data set
function [features, classes] = loadDataSet(dataSetNo)
    switch dataSetNo
        case 1
            disp('WBCD Data set: Features: 30, Classes: 2');
            dataSet = importdata('WBCD.data');
            features = dataSet.data;
            classes = cell2mat(dataSet.textdata(:,2))-'A';% char to numeric
        case 2
            disp('Sonar Data set: Features: 60, Classes: 2');
            dataSet=readtable('sonar.data', 'ReadVariableNames', false, 'FileType', 'delimited')
            dataSet.Var61=cellfun(@(v) v-'A', dataSet.Var61);
            dataSet=table2array(dataSet);
            features = dataSet(:, 1:end-1);
            classes = dataSet(:, end);
        case 3
            disp('Movement_libras Data set: Features: 90, Classes: 15');
            dataSet = importdata('movement_libras.data');
            features = dataSet(:, 1:end-1);
            classes = dataSet(:, end);
        case 4
            disp('Hill_Valley Data set: Features: 100, Classes: 2');
            trainData = importdata('Hill_Valley_without_noise_Training.data').data;
            testData = importdata('Hill_Valley_without_noise_Testing.data').data;
            dataSet = vertcat(trainData, testData);
            features = dataSet(:, 1:end-1);
            classes = dataSet(:, end);
        case 5
            disp('Musk1 Data set: Features: 166, Classes: 2');
            file = importdata('musk.csv');
            dataSet = file.data;
            features = dataSet(:, 4:end-1);
            classes = dataSet(:, end);
        case 6
            disp('Mfeat-pix Data set: Features: 240, Classes: 10');
            file = importdata('mfeat-pix');
            features = file;
            % No labels available for this data set, its just ordered with 200 samples per class
            classes = [zeros(200,1); ones(200,1); 2*ones(200,1); 3*ones(200,1); 4*ones(200,1); 5*ones(200,1); ...
                6*ones(200,1); 7*ones(200,1); 8*ones(200,1); 9*ones(200,1)];
        case 7
            disp('Arrhythmia Data set: Features: 279, Classes: 16');
            dataSet = readtable('arrhythmia.data', 'ReadVariableNames', false, 'FileType', 'delimited');
            % There are strings in the data set, so we need to convert them to numbers
            dataSet.Var14=cellfun(@str2double, dataSet.Var14);
            dataSet=table2array(dataSet);
            % Replace NaN with 0
            dataSet(isnan(dataSet))=0;
            features = dataSet(:, 1:end-1);
            classes = dataSet(:, end);
        case 8
            disp('Madelon Data set: Features: 500, Classes: 2');
            file = importdata('madelon.csv');
            dataSet = file.data;
            features = dataSet(:, 1:end-1);
            classes = dataSet(:, end);
        otherwise
            assert(false, 'Invalid data set number');
    end
    disp('-----------------------------------------------');
end

% Normalization (min-max normalization)
function Data = normalizeData(Data)
    Fmin = min(Data(:,1:end-1),[],1);
    Fmax = max(Data(:,1:end-1),[],1);
    Data(:,1:end-1) = (Data(:,1:end-1)-repmat(Fmin,size(Data,1),1))./repmat(Fmax-Fmin,size(Data,1),1);
end

