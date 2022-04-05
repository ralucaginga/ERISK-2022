# ERISK-2022

### Notes
Reddit post blueprint for time series:
 - from a stack classifier by extracting just one model output
 - How about a sentiment analysis ?
 
 There are 2 metodologies for training:

 1) Individual Model
    Deals with a twit as input and outputs a probability (confidence) for the for the user to be depressed or in control.
    
 2) Time Series Model
    Deals with the output probability of the the individual model for a set of twits. The twits are selected using a rolling window of a fixed size and a gap as a stride.

Using the label of the eRisk dataset can be dummy because we don't have a timestamp (or a twit index) for the decising message that can provide a diagnostic, along with a window_size historic.