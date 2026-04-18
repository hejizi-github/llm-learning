# Sample: Karpathy — "The Unreasonable Effectiveness of RNNs" (2015)
Source: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
Fetched: 2026-04-19

---

There's something magical about Recurrent Neural Networks (RNNs). I still remember when I
trained my first recurrent network for Image Captioning. Within a few dozen minutes of
training my first baby model (with rather arbitrarily-chosen hyperparameters) started to
generate very nice looking descriptions of images that were on the edge of making sense.
Sometimes the ratio of how simple your model is to the quality of the results you get out
of it blows past your expectations, and this was one of those times. What made this result
so shocking at the time was that the common wisdom was that RNNs were supposed to be
difficult to train (with more experience I've in fact reached the opposite conclusion).
Fast forward about a year: I'm training RNNs all the time and I've witnessed their power
and robustness many times, and yet their magical outputs still find ways of amusing me.
This post is about sharing some of that magic with you.

"We'll train RNNs to generate text character by character and ponder the question
'how is that even possible?'"
