clc;
clear;
close all;
    % Load data
    dataNo = 4;

    switch dataNo
        case 1
            disp('Hill_Valley');
            [TrainIn, TrainOut] = prepareData('Hill_Valley_without_noise_Training.data');
            [ValidIn, ValidOut] = prepareData('Hill_Valley_without_noise_Testing.data');
            
        case 2
            disp('musk');
            file = importdata('musk.csv');
            Data = file.data;
            Category = unique(Data(:,end));

        case 3
            disp('madelon');
            file = importdata('madelon.csv');
            fullDataSet = file.data;
            Categor = unique(fullDataSet(:,end));

        case 4
            disp('movement_libras');
            data = importdata('movement_libras.data');
            Classes = unique(data(:,end));

    end

    Category = unique(TrainOut(:,end));
    

    function [input, output] = prepareData(dataFileName)
        file = importdata(dataFileName);
        Data = file.data;
        Fmin = min(Data(:,1:end-1),[],1);
        Fmax = max(Data(:,1:end-1),[],1);
        Data(:,1:end-1) = (Data(:,1:end-1)-repmat(Fmin,size(Data,1),1))./repmat(Fmax-Fmin,size(Data,1),1);
        input = Data(:,1:end-1);
        output = Data(:,end);
    end
