# Multimodal-DialogueGCN

This repository contains the code accompanying the MSc dissertation at the Computer Science Department of UCL.

For the results as included in the written report, see results folder that contains a jupyter notebook version of the code, with the results and outputs. 


This work introduces a multimodal approach to the DialogueGCN originally presented in:
DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. D. Ghosal, N. Majumder, S. Poria, N. Chhaya, & A. Gelbukh. EMNLP-IJCNLP (2019), Hong Kong, China.
This is done by utilisation of four different fusion mechanisms: early fusion, late fusion, decision-level fusion, and fusion via External Context Encoder (ECE).
Fusion via ECE is a novel fusion approach that incorporates information from additional modalities through a GRU-based sub-model capturing the contextual information included in new modalities.


Thic code utilises some classes of the open-sourced code licensed under MIT license by declare-lab, available at https://github.com/declare-lab/conv-emotion. 
