# vanilla-LSTM

### Run this thing on bash by <code>bash run_this_thing_already.sh</code>

This repository builds, trains and LSTM on, and predicts on a bunch of vectors of random data with monotonically increasing trailing digits. The network predicts the last digit of an unseen number which would come next in sequence. For example the network will see a vector like

31,542,883,4,15

and it will predict the following sequence:

2,3,4,5,6

where, obviously the 6 would come next.<br><br>

It's a pretty silly little RNN which uses a little single layer, 500 neuron, LSTM. The project was just a way to do something a little different than the zillions of "next character in a sequence predictions" you see all the time. Anyway, I've used <a href="https://github.com/Element-Research/rnn" target="_blank">Element Research's <code>RNN</code></a> code to build it. You should check out their toolkit if you haven't seen it; it's quite nice.
