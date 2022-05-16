Here，we describe how to run the codes.

A total of three steps are required：

Step 1:
Create the Pool Set and Testing Set with five times five-fold Stratified Cross-validation. 
Using CreatPartition_65SCV.py in "Partition_Pool_Testing"

Step 2:
Active instances selection by running the compared algorithms in "Compared_Methods".

Step 3:
Run KELMOR on the selected results.
Using "Run_Classifier_on_Selected_Results.py"