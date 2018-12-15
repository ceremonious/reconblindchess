# Reconnaissance Chess

Working towards making an AI agent that can play Recon chess. Rules for the game are described [here](http://rbmc-ext.jhuapl.edu/).

## Overview

So far, the progress has been focused on making a model that can predict the board state of the board given observations that are made through sensing (looking at 3x3 section of the board) or moving/captures/being captured. A board state is represented as a 8x8x13 array. Each square on the board has a corresponding array of 13, which is a probability distribution for what piece could be on that square. On each square, there are 13 possibilities (one of my 6 pieces, one of their 6 pieces or an empty square).

An observation is also modeled as an 8x8x13 array. For each square, if the player learns that a certain piece is on there, or that it is empty, then that possibility will be indicated with a 1. If nothing about that square is being observed, then all 13 values will be 0.

## Training

Currently, the model is trained by having the computer continually play against itself. During each turn, two observations are made. One observation is made before the start of the turn that contains any information gained since the last move as well as information about where the player's pieces are located. Another observation is made after sensing, which gives the player information about a 3x3 section of the board. Every time an observation is made, the true board state at that time is stored. At the end of the game, a sequence of observations and a sequence of corresponding true board states are fed into the model as training examples

## Files

* reconBoard.py - extension of the [python-chess](https://github.com/niklasf/python-chess) library that supports recon blind chess
* models.py - contains Keras code for creating policy/belief state neural net
* belief_state_training.py - trains belief state neural net
* play.py - tests belief state neural net
