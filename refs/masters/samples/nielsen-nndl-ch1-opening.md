# Sample: Michael Nielsen — "Neural Networks and Deep Learning" Chapter 1 (2019)
Source: http://neuralnetworksanddeeplearning.com/chap1.html
Fetched: 2026-04-19

---

## Opening

The human visual system is one of the wonders of the world. Consider the following
sequence of handwritten digits: Most people effortlessly recognize those digits as 504192.
That ease is deceptive. In each hemisphere of our brain, humans have a primary visual
cortex, also known as V1, containing 140 million neurons, with tens of billions of
connections between them. And yet human vision involves not just V1, but an entire series
of visual cortices - V2, V3, V4, and V5 - doing progressively more complex image
processing. We carry in our heads a supercomputer, tuned by evolution over hundreds of
millions of years, and superbly adapted to understand the visual world. Recognizing
handwritten digits isn't easy. Rather, we humans are stupendously, astoundingly good at
making sense of what our eyes show us. But nearly all that work is done unconsciously.

The difficulty of visual pattern recognition becomes apparent if you attempt to write a
computer program to recognize digits like those above. What seems easy when we do it
ourselves suddenly becomes extremely difficult. Simple intuitions about how we recognize
shapes — "a 9 has a loop at the top, and a vertical stroke in the bottom right" — turn
out to be not so simple to express algorithmically. When you try to make such rules
precise, you quickly get lost in a morass of exceptions and caveats and special cases.
It seems hopeless.

Neural networks approach the problem in a different way. The idea is to take a large
number of handwritten digits, known as training examples, and then develop a system which
can learn from those training examples. In other words, the neural network uses the examples
to automatically infer rules for recognizing handwritten digits.

In this chapter we'll write a computer program implementing a neural network that learns to
recognize handwritten digits. The program is just 74 lines long, and uses no special neural
network libraries. But this short program can recognize digits with an accuracy over 96
percent, without human intervention.

## Perceptrons section (how to introduce a technical concept)

What is a neural network? To get started, I'll explain a type of artificial neuron called
a **perceptron**. Perceptrons were developed in the 1950s and 1960s by the scientist Frank
Rosenblatt, inspired by earlier work by Warren McCulloch and Walter Pitts.

So how do perceptrons work? A perceptron takes several binary inputs, x1, x2, ..., and
produces a single binary output.

Rosenblatt proposed a simple rule to compute the output. He introduced **weights**,
w1, w2, ..., real numbers expressing the importance of the respective inputs to the output.
The neuron's output, 0 or 1, is determined by whether the weighted sum is less than or
greater than some **threshold value**.

That's all there is to how a perceptron works! That's the basic mathematical model.

A way you can think about the perceptron is that it's a device that makes decisions by
weighing up evidence. Let me give an example. It's not a very realistic example, but it's
easy to understand, and we'll soon get to more realistic examples. Suppose the weekend is
coming up, and you've heard that there's going to be a cheese festival in your city. You
like cheese, and are trying to decide whether or not to go to the festival. You might make
your decision by weighing up three factors:
- Is the weather good?
- Does your boyfriend or girlfriend want to accompany you?
- Is the festival near public transit? (You don't own a car).
