import time
import numpy as np
import scipy.stats
import random
import pygame
import os
import sys
Ranks = ['1', '2', '3', '4',
                 '5', '6', '7', '8',
                 '9', '10', '11', '12', '13']
Suits = ['S', 'C', 'H', 'D']
Deck = []

pygame.font.init()
font = pygame.font.Font("C:/Users/peter/Downloads/PixeloidSans-mLxMm.ttf", 18)
# Render the text
text = "Player 1 Num Cards"


def check_slappable(deck,i):
    slappable = False
    top_card = deck[i]

    if i <=49:
        if deck[i+2][:-1] == top_card[:-1]:
            slappable = True
            # print('Slappable Sandwich')
    if i<=50:
        if deck[i+1][:-1] == top_card[:-1]:  # two in a row
            slappable = True
        # if top_card[:-1] not in ['1', '13']:  # consecutives
        #     diff = int(deck[i+1][:-1]) - int(top_card[:-1])
        #     if diff == 1 or diff == -1:
        #         slappable = True
        # if top_card[:-1] == '13':
        #     if deck[i+1][:-1] == '1' or deck[i+1][:-1] == '12':
        #         slappable = True
        # if top_card[:-1] == '1':
        #     if deck[i+1][:-1] == '2' or deck[i+1][:-1] == '13':
        #         slappable = True
    return slappable

for rank in Ranks:
    for suit in Suits:
        Deck.append(rank + suit)


def run_sim(trials):
    slappable_lst = []

    for i in range(trials):
        random.shuffle(Deck)
        # print(Deck)
        Deck_sub = Deck.copy()
        len_Deck = len(Deck)
        lst = []
        num = 0
        for i in range(len_Deck):
            if check_slappable(Deck,i):
                num+=1
            lst.append(check_slappable(Deck,i))
        slappable_lst.append(num)
    return slappable_lst

result = run_sim(1000)
print(result)
print('avg', np.average(result))
print('var', np.var(result))


