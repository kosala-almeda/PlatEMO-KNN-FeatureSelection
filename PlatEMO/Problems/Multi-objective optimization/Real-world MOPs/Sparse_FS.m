classdef Sparse_FS < PROBLEM
% <multi> <binary> <large/none> <expensive/none> <sparse/none>
% The feature selection problem for KNN classification
% dataNo --- 1 --- Data set number (1-Musk1, 2-Semeion_handwritten_digit, 3-LSVT_voice_rehabilitation)

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

% The datasets are taken from the UCI machine learning repository in
% http://archive.ics.uci.edu/ml/index.php
% No.   Name                              Samples Features Classes
% 1     MUSK1                               476     166       2
% 2     Semeion_handwritten_digit          1593     256      10
% 3     LSVT_voice_rehabilitation           126     310       2

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
            dataNo = obj.ParameterSet(1);
            str    = {'MUSK1','Semeion_handwritten_digit','LSVT_voice_rehabilitation'};
            CallStack = dbstack('-completenames');
            load(fullfile(fileparts(CallStack(1).file),'Dataset_FS.mat'),'Dataset');
            Data = Dataset.(str{dataNo});
            % Shuffle the dataset
            Data = Data(randperm(size(Data,1)),:);
            % Normalization (min-max normalization)
            Fmin = min(Data(:,1:end-1),[],1);
            Fmax = max(Data(:,1:end-1),[],1);
            Data(:,1:end-1) = (Data(:,1:end-1)-repmat(Fmin,size(Data,1),1))./repmat(Fmax-Fmin,size(Data,1),1);
            % Classes of the dataset
            obj.Category    = unique(Data(:,end));
            % Divide the dataset into training, validation and test sets
            %  training set: 50%
            obj.TrainIn     = Data(1:ceil(end*0.5),1:end-1);
            obj.TrainOut    = Data(1:ceil(end*0.5),end);
            %  validation set: 20%
            obj.ValidIn     = Data(ceil(end*0.5)+1:ceil(end*0.7),1:end-1);
            obj.ValidOut    = Data(ceil(end*0.5)+1:ceil(end*0.7),end);
            %  test set: 30%
            obj.TestIn     = Data(ceil(end*0.7)+1:end,1:end-1);
            obj.TestOut    = Data(ceil(end*0.7)+1:end,end);
            % Parameter setting
            obj.M        = 2; % Number of objectives (Number of selected features, Validation error)
            obj.D        = size(obj.TrainIn,2); % Number of decision variables (Number of features in the dataset)
            obj.encoding = 4 + zeros(1,obj.D); % Encoding design (4: binary encoding)
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
        %% Display a population with post optimization test results
        function DrawTest(obj,Population)
            try
                PopObj = Population.adds;
            catch
                obj.PostOptimization(Population);
                PopObj = Population.adds;
            end
            PopObj(:,1) = PopObj(:,1)*obj.D;
            PopObj(:,2) = PopObj(:,2)*100;
            Draw(PopObj,{'No. of selected features','Test error %',[]});
        end
        %% Calculate the error for test set
        function PostOptimization(obj,Population)
            disp('PostOptimization')
            PopDec = logical(Population.decs);
            for i = 1 : size(PopDec,1)
                % Rank the training samples according to their distances to the current solution
                [~,Rank] = sort(pdist2(obj.TestIn(:,PopDec(i,:)),obj.TrainIn(:,PopDec(i,:))),2);
                % Predict the labels by the majority voting of K(=5) Nearest Neighbors
                %   mode is not used as following strategy is better at tie breaking
                %   it gives priority close neighbors in case of a tie
                [~,Out]  = max(hist(obj.TrainOut(Rank(:,1:5))',obj.Category),[],1); 
                Out      = obj.Category(Out);
                % Using mean over sum to normalize the objective value between 0 and 1
            	PopTest(1, 1) = mean(PopDec(i,:));
                % Validation error is the ratio of misclassified samples
                PopTest(1, 2) = mean(Out~=obj.TestOut);
                % Store in the population
                Population(i).add = PopTest;
            end
        end
    end
end