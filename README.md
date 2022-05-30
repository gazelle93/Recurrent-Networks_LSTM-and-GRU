# Overview
- Recurrent-Neural-Networks (RNN) is the sequential deep learning architecture that allows the previous node's outputs to be used with the current node's inputs when computing the current node's output so the current node's output is enabled to be influenced by not only the current node's input but the previous node's output. While the feed-forward network parameters that belong to each node in the same layer are different. However, it is difficult to handle information that has a long sequence due to vanishing gradient and exploding gradient issues. This problem can be solved by replacing activation functions or using variation of RNN model such as Long-Short-Term-Memory (LSTM) and Gated-Recurrent-Unit (GRU). LSTM has a special component called forget gate which can decide whether to preserve all the information so far or forget everything. GRU is the simpler version of LSTM. This project aims to implement LSTM and GRU.

- LSTM:
$$i_t = \sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi})$$
$$f_t = \sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf})$$
$$g_t = \tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg})$$
$$o_t = \sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho})$$
$$c_t=f_t\odot c_{t-1} + i_t\odot g_t$$
$$h_t=o_t\odot \tanh(c_t)$$

where $x_t$ is word embedding at time $t$, $h_t$ is the hidden state at time $t$, $c_t$ is the cell state at time $t$ and $i_t$, $f_t$, $g_t$, $o_t$ are the input, forget, cell and output gates respectively. 

- GRU:
$$r_t = \sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr})$$
$$z_t = \sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz})$$
$$n_t = \tanh(W_{in}x_t+b_{in}+r_t*(W_{hn}h_{t-1}+b_{hn}))$$
$$h_t=(1-z_t) * n_t + z_t * h_{t-1}$$

where $x_t$ is word embedding at time $t$, $h_t$ is the hidden state at time $t$, and $r_t$, $z_t$, $n_t$ are the reset, update and new gates respectively.

# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- my_onehot.py
> Output format
> - output: List of tensor of input tokens. (Tensor)
- lstms.py
> Output format
> - output: List of tensor of lstm/gru results. (Tensor)


# Prerequisites
- argparse
- torch
- stanza
- spacy
- nltk
- gensim

# Parameters
- nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
- unk_ignore(bool, defaults to True): Ignore unseen word or not.
- bidirection(bool, defaults to False): Use of bidirectional lstm/gru.
- sequential_model(str, defaults to "lstm"): Type of sequential model layer. (lstm, gru)

# References
- RNN: Elman, J. L. (1990). Finding structure in time. Cognitive science, 14(2), 179-211.
- LSTM: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- GRU: Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python. O'Reilly Media Inc.
- Gensim: Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In In Proceedings of the LREC 2010 workshop on new challenges for NLP frameworks.
