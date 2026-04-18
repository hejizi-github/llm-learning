# Sample: Karpathy — "Hacker's Guide to Neural Networks" (2014)
Source: https://karpathy.github.io/neuralnets/
Fetched: 2026-04-19

---

My personal experience with Neural Networks is that everything became much clearer when I
started ignoring full-page, dense derivations of backpropagation equations and just started
writing code. Thus, this tutorial will contain very little math (I don't believe it is
necessary and it can sometimes even obfuscate simple concepts). Since my background is in
Computer Science and Physics, I will instead develop the topic from what I refer to as
hackers's perspective. My exposition will center around code and physical intuitions instead
of mathematical derivations. Basically, I will strive to present the algorithms in a way
that I wish I had come across when I was starting out.

"…everything became much clearer when I started writing code."

You might be eager to jump right in and learn about Neural Networks, backpropagation, how
they can be applied to datasets in practice, etc. But before we get there, I'd like us to
first forget about all that. Let's take a step back and understand what is really going on
at the core. Lets first talk about real-valued circuits.
