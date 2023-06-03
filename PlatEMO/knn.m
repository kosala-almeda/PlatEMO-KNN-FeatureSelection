%Main index file


% https://github.com/amoudgl/kNN-classifier


clear;
close all;

tic

filename = 'iris.data.txt';
A = dataset('File', filename, 'Delimiter', ',', 'ReadVarNames', false);

%Assigning classes a unique number for conversion from dataset to matrix
%Even if classes are more in number with randomly arranged datapoints, this method would work
u = unique(A(:, 5));
u = dataset2cell(u);
u = u(2 : end, :);
p = [1 : size(u, 1)];
C = containers.Map(u, p);
t = A(:, 5);
t = dataset2cell(t);
t = t(2 : end);
A.Var5 = C.values(t);
z = A(:, 5);
z = dataset2cell(z);
z = z(2 : end);
z = cell2mat(z);
z = mat2dataset(z);
A(:, 5) = z;
A = double(A);

%Generating random permutation for division into test and train data
nrows = size(A, 1);
randrows = randperm(nrows);

%meanaccuracy contains the accuracy data
%rows in meanaccuracy denote fold
%and column denote the corresponding value of K

% meanaccuracy = zeros(4, 5);
% X = A;
% k = [2, 3, 4, 5];
% e = zeros(4, 1);
% for K = 1 : 5
%     for fold = 2 : 5
%         for chunk = 1 : fold
%             chunksize = floor(nrows/fold);
%             x = (chunk - 1) * chunksize + 1;
%             y = chunk * chunksize;
%             testdata = X(randrows(x:y), :);
%             if chunk == 1
%                 traindata = X(randrows(y + 1:end), :);
%             elseif chunk == fold
%                 traindata = X(randrows(1 : x-1), :);
%             else
%                 traindata = X(randrows(1, x-1:y+1, end), :);
%             end
%             currentacc = knnclassifier(traindata, testdata, K);
%             s(chunk) = currentacc;
%         end
%         meanaccuracy(fold - 1, K) = mean(s);
%         out(fold - 1) = mean(s);
%         e(fold - 1) = std(s);
%     end
%     subplot(3,3, K);
%     errorbar(k, out, e);
%     title(['Plot for K = ', num2str(K)])
% end

traindata = A(randrows(1 : ceil(0.7 * nrows)), :);  % use first 70% of data for training
% use last 30% of data for testing
testdata = A(randrows(ceil(0.7 * nrows) : end), :);

%KNN Classifier function call
% [accur,class] = knnclassifier(traindata, testdata, 3);
% use Problems/Multi-objective optimization/KNN-Features/knnclassifier.m
addpath('Problems/Multi-objective optimization/KNN-Features');
knnc = KNNClassifier(3);
kfold = KFoldValidator(4, knnc);
accur = kfold.validate(traindata(:, 1 : 4), traindata(:, 5));

display(['Accuracy for K = 3 is ', num2str(accur * 100), '%'])
% knnc = knnc.fit(traindata(:, 1 : 4), traindata(:, 5));
% class = knnc.predict(testdata(:, 1 : 4));

% % calculate accuracy
% accur = sum(class == testdata(:, 5)) / size(testdata, 1);

% disp(['Accuracy for K = 3 is ', num2str(accur * 100), '%'])

% % plot classification results
% figure;

% % first two dimensions of data are plotted
% subplot(1, 2, 1);
% hold on;
% plot(testdata(testdata(:, 5) == 1, 1), testdata(testdata(:, 5) == 1, 2), 'r.');
% plot(testdata(testdata(:, 5) == 2, 1), testdata(testdata(:, 5) == 2, 2), 'g.');
% plot(testdata(testdata(:, 5) == 3, 1), testdata(testdata(:, 5) == 3, 2), 'b.');
% plot(testdata(class == 1, 1), testdata(class == 1, 2), 'ro');
% plot(testdata(class == 2, 1), testdata(class == 2, 2), 'go');
% plot(testdata(class == 3, 1), testdata(class == 3, 2), 'bo');
% title('Classification results');
% subtitle(sprintf('Accuracy = %f', accur));
% xlabel('Sepal length');
% ylabel('Sepal width');
% legend('Iris-setosa', 'Iris-versicolor', 'Iris-virginica');
% % dot is actual class, circle is predicted class
% hold off;

% % last two dimensions of data are plotted
% subplot(1, 2, 2);
% hold on;
% plot(testdata(testdata(:, 5) == 1, 3), testdata(testdata(:, 5) == 1, 4), 'r.');
% plot(testdata(testdata(:, 5) == 2, 3), testdata(testdata(:, 5) == 2, 4), 'g.');
% plot(testdata(testdata(:, 5) == 3, 3), testdata(testdata(:, 5) == 3, 4), 'b.');
% plot(testdata(class == 1, 3), testdata(class == 1, 4), 'ro');
% plot(testdata(class == 2, 3), testdata(class == 2, 4), 'go');
% plot(testdata(class == 3, 3), testdata(class == 3, 4), 'bo');
% title('Classification results');
% subtitle(sprintf('Accuracy = %f', accur));
% xlabel('Petal length');
% ylabel('Petal width');
% % show in legend that dot is actual class, circle is predicted class
% legend('Iris-setosa', 'Iris-versicolor', 'Iris-virginica');
% hold off;


toc


% KNN Classifier function

function  [accur, expclass] = knnclassifier(traindata, testdata, K)

    % Find distance with all training datapoints, sort and poll
    for i = 1 : size(testdata)
        x = testdata(i,:);
        
        % Euclidean distance calculation
        % dist = sqrt((traindata(:, 1) - x(1)) .^ 2 + (traindata(:, 2) - x(2)) .^ 2 + (traindata(:, 3) - x(3)) .^ 2 + (traindata(:, 4) - x(4)) .^ 2);
        
        % Manhattan distance calculation
        dist = abs(traindata(:, 1) - x(1)) + abs(traindata(:, 2) - x(2)) + abs(traindata(:, 3) - x(3)) + abs(traindata(:, 4) - x(4));
        classes = traindata(:, 5);
        dist(:, 2) = classes;
        poll = sortrows(dist, 1);
        
        expclass(i) = mode(poll(1 : K, 2));
        
    end
    
    % Error percentage calculation
    error = transpose(expclass) - testdata(:,5);
    accur = ((size(error, 1) - nnz(error))/size(error, 1));

end