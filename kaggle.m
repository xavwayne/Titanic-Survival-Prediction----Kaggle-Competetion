%Decision-tree model for kaggle competition
close all;
clear;
clc

%load and parse input file

filename = 'train.csv';
delimiter = ',';
formatSpec = '%d%d%d%q%s%f%d%d%s%f%s%s';
fileID = fopen(filename,'r'); % Open the text file.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'headerLines', 1);
fclose(fileID); % Close the text file.

properties = { 'PassengerId', 'Survived', 'Pclass', 'Name', ...
    'Sex','Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked' };

titanic=dataset(dataArray{:});
titanic.Properties.VarNames = properties;
titanic.Sex = grp2idx(cellstr(titanic.Sex));
idx= ~isnan(titanic.Age) & ~isnan(titanic.Pclass) & ~isnan(titanic.Sex);
y = titanic.Survived(idx);
%select important features
X =[double(titanic.Pclass(idx)),double(titanic.Sex(idx)),double(titanic.Age(idx))];

%create decission tree
ctree = ClassificationTree.fit(X,y);

% view the decision tree via its rules
view(ctree)

% create a visual graphic for the tree
view(ctree,'mode','graph')

% Pruning a Classification Tree
% Find the optimal pruning level by minimizing cross-validated loss:
[~,~,~,bestlevel] = cvLoss(ctree,'subtrees','all','treesize','min')

% Prune the tree to use it for other purposes:
pctree = prune(ctree,'Level',bestlevel);
view(pctree,'mode','graph')

%perform prediction on the test set
%load and parse the input test file
filename = 'test.csv';
delimiter = ',';
formatSpec = '%d%d%q%s%f%d%d%s%f%s%s';
fileID = fopen(filename,'r'); % Open the text file.
dataArray_test = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'headerLines', 1);
fclose(fileID); % Close the text file.

properties = { 'PassengerId', 'Pclass', 'Name', ...
    'Sex','Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked' };

test = dataset(dataArray_test{:});
test.Properties.VarNames = properties;
test.Sex = grp2idx(cellstr(test.Sex));
%select important features
X_test = [double(test.Pclass),double(test.Sex),double(test.Age)];
%make a prediction
y_test = predict(pctree, X_test);
%output the result
outfile = 'output.csv';
outID = fopen(outfile, 'w');
fprintf(outID, 'PassengerId,Survived\n');
fclose(outID);
dlmwrite(outfile, [test.PassengerId y_test], '-append', 'delimiter', ',');




