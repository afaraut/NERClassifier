# NERClassifier

`###############################################################################`
##		Project : Regression NER Classifier
##		Author : Anthony FARAUT
`###############################################################################`

# Files : 
	- generate_data.py : Generate and store the data features.
	- arguments_parser.py : Parse the arguments from the command line.
	- ner.py : Name entity recognition Classifier.

# Data :
	/in
	- Files provided for the project
	/out
	- Files generated with the features
	/train
	- Logistic regression model saved

# Features :
	- 1 - The number of vowel;
	- 2 - The number of consonant;
	- 3 - If the word is capitalized (first letter);
	- 4 - If the word is a name;
	- 5 - If the word is a number;
	- 6 - The size of the word;
	- 7 - If the word is capitalized (all the letters);
	- 8 - If the word is capitalized (first letter) and the previous word is a end punctuation;
	- 9 - If the word is a end punctuation.

# Example of use : 

	python ner.py -train ..\data\out\ned_train.csv

	python ner.py -test ..\data\out\ned_testa.csv -learning_curve
	Precision  0.874495848161
	Recall  0.880343921662
	F1  0.877410140443

	python ner.py -test ..\data\out\ned_testb.csv -learning_curve
	Precision  0.867724867725
	Recall  0.870092466273
	F1  0.868907054193

	python ner.py -train ..\data\out\esp_train.csv

	python ner.py -test ..\data\out\esp_testa.csv -learning_curve
	Precision  0.818794506612
	Recall  0.955624814485
	F1  0.881933981646

	python ner.py -test ..\data\out\esp_testb.csv -learning_curve
	Precision  0.807783204664
	Recall  0.967723669309
	F1  0.8805495921
