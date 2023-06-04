classdef NSGAII < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained/none>
% Nondominated sorting genetic algorithm II
% unique --- 0 --- Extremely diversity. Preserve only unique solutions (0-disabled, 1-parents, 2-offspring, 3-both)

%------------------------------- Reference --------------------------------
% K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, A fast and elitist
% multiobjective genetic algorithm: NSGA-II, IEEE Transactions on
% Evolutionary Computation, 2002, 6(2): 182-197.
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

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);

            %% Optimization
            while Algorithm.IsNotTerminated(Population)
                % GA operations (crossover and mutation)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool), {0.9, 20, 0.01, 20});

                % Remove duplicate solutions for extreme diversity
                % Prioritizes uniqueness over dominance
                uniquePopulation = [Population, Offspring];
                if Algorithm.ParameterSet(1) == 1 % Parent unique selection
                    [~,uniqueIndex] = unique(Population.decs,'rows');
                    uniquePopulation = [Population(uniqueIndex), Offspring];
                elseif Algorithm.ParameterSet(1) == 2 % Offspring unique selection
                    [~,uniqueIndex] = unique(Offspring.decs,'rows');
                    uniquePopulation = [Population, Offspring(uniqueIndex)];
                elseif Algorithm.ParameterSet(1) == 3 % Both unique selection
                    [~,uniqueIndex] = unique(uniquePopulation.decs,'rows');
                    uniquePopulation = uniquePopulation(uniqueIndex);
                    % Make sure the population size is maintained
                    if length(uniquePopulation) < Problem.N
                        % add random from remaining population
                        remainingIndexes = setdiff(1:length(Population), uniqueIndex);
                        randomIndexes = randperm(length(remainingIndexes), Problem.N - length(uniquePopulation));
                        uniquePopulation = [uniquePopulation, Population(remainingIndexes(randomIndexes))]; %#ok<AGROW>
                    end
                end

                % Non-dominated sorting
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection(uniquePopulation,Problem.N);
            end
        end

        %% Non throwing termination criteria
        function bool = IsNotTerminated(Algorithm,Population)
            try
                bool = Algorithm.NotTerminated(Population);
            catch err
                % check if error is termination assertion error
                if strcmp(err.identifier, 'PlatEMO:Termination')
                    bool = false;
                else
                    rethrow(err);
                end
            end
        end
    end
end