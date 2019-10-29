:::::::::::::::::::::::::::::::::
:: SEA Dataset ::::::::::::::::::
:::::::::::::::::::::::::::::::::


See dataset description in: 

W. N. Street and Y. Kim, "A streaming ensemble algorithm (SEA) for 
 large-scale classification," in Seventh ACM SIGKDD International 
 Conference on Knowledge Discovery & Data Mining (KDD-01) 2001, 
 pp. 377-382.

::: Dataset Information ::::::::::::::::::::::::::::::::::::::
 Features: 3
 Classes: 2
 Time steps: 200
 Training instances per time step: 250
 Testing instances per time step: 250
 Total Training Instances: 50,000
 Total Testing Instances: 50,000


::: Training/Testing Procedure (Batch Learning) ::::::::::::::

 At each time step, train on a window of 250 samples from file
 "SEA_training_data.csv" (50,000 x 3) with associated class labels in file
 "SEA_training_class.csv" (50,000 x 1) After training, you may test with 
 a window of size 250 from file "SEA_testing_data.csv" (50,000 x 3) 
 with associated class labels in file "SEA_testing_class.csv" (50,000 x 3).  
 For subsequent time steps, shift the training and testing windows by 
 250 and read in the next batch.



==========================================================
Ryan Elwell, Robi Polikar
Signal Processing & Pattern Recognition Laboratory (SPPRL)
Department of Electrical & Computer Engineering
Rowan University
==========================================================