# Enron_person_of_interest
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.

Through this project, I hope to identify more persons of interest with the given database of email conversations, financial information of the stock options and the existing known persons of interest.

poi_id.py : starter code for the POI identifier 

final_project_dataset.pkl : the dataset for the project

tester.py : To test the result, to make sure we see performance that’s similar to what you report. 

emails_by_address : this directory contains many text files, each of which contains all the messages to or from a particular email address. 

POI_ID.PY:
1. I selected what features to use.features_list is a list of strings, each of which is a feature name.
2. Removed outliers which were data discrepancies and anomalies.
3. Created new feature and stored to my dataset.
4. Ran some features selection and dimensionality reduction algorithms to this data.
5. Used the supervised algorithm- Decision trees to fit this data and evaluated using cross validation.

