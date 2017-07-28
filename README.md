Optional arguments when running backprop_hidden_units: use -u to specify the number of hidden units, 
use -r to specify the number of restarts. Default number of hidden units is 1 and default restarts is 100. 
If you run -u 10 -r 10 then the code will randomly generate weight matrices for a 10 unit single layer, single character RNN,
it will do this 10 times and then will output the learned weights that give the highest objective. 

Near the top of the file, you can specify the characters - currently set to 'a' and '$' where '$' is used as the string boundary
symbol. This can be changed and the code will still work. We also specify the 'data' here which is the training set.

The variable con_length can be changed which is -1 if we want the function 'consistency' to calculate the total probability of 
all finite strings, and if it's >0 'consistency' calculates the total probability of all strings up to length con_length.

