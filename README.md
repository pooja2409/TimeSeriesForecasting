# TimeSeriesForecasting
Time Series Forecasting using XGBoost Classifier
Technical Details:
<br />
Linux:Ubuntu 16.04
<br />
Language used: Python
<br />
Framework:scikit-learn
<br />
Qn1: Briefly describe the conceptual approach you chose! What are the trade-offs?<br />
#Conceptual Approach:<br />
    1. Read The data through python Pandas.<br />
    2. Cleaning the Data.<br />
    3.Analysing the Data by plotting a graph.<br />
    4.Changing the Timestamp column of the dataframe to year, month, day, minutes, hour, second separate columns.<br />
    5.Fitting the model in a XGBoost Classifier for prediction.<br />
    6.Predicting the output of the test data.<br />
    7.Plotting the time series test data.<br />
<br />
#Trade Off:<br />
GBTs build trees one at a time, where each new tree helps to correct errors made by previously trained tree. With each tree added, the model becomes even more expressive. There are typically three parameters - number of trees, depth of trees and learning rate, and the each tree built is generally shallow. 
<br />

Qn2: What's the model performance?<br />
The accuracy on the test split of the training dataset is 99.15%.
<br />
QN3: What's the runtime performance? What is the complexity? Where are the bottlenecks?<br />
The original sparse greedy algorithm doesn't use block storage.
Thus to find the optimal split at each node, you needed to re-sort the data on each column.
This ends up incurring a time complexity at each layer that is very crudely approximated by O(‖x‖0logn):
basically, say you have ‖x‖0i nonzero entries for each feature 1≤i≤m;
then at each layer you're sorting lists, each of length at most n, whose lengths sum to ∑mi=1‖x‖0i=‖x‖0
which can't take more than O(‖x‖0logn) time.
Multiplying by K trees and d layers per tree gives you the original O(Kd‖x‖0logn) time complexity.
The most time consuming part of the tree learning algorithm is getting the data in sorted order.
This makes the time complexity of learning each tree O(n log n).<br />
<br />
<br />
Qn4: If you had more time, what improvements would you make, and in what order of priority?<br />
If given more time I would have used *Multivariate LSTM* and Time-Series-ARIMA-XGBOOST-RNN
Include the features per timestamp Sub metering 1, Sub metering 2 and Sub metering 3, date, time and our target variable into the RNNCell for the multivariate time-series LSTM model.
<br />
