import time
import numpy as np
import scipy.stats
import random
import pygame
import os
import sys
random.seed(1111)
Ranks = ['1', '2', '3', '4',
                 '5', '6', '7', '8',
                 '9', '10', '11', '12', '13']
Suits = ['S', 'C', 'H', 'D']
Deck = []

pygame.font.init()
font = pygame.font.Font("C:/Users/peter/Downloads/PixeloidSans-mLxMm.ttf", 18)
# Render the text
text = "Player 1 Num Cards"


for rank in Ranks:
    for suit in Suits:
        Deck.append(rank + suit)



class player():
    def __init__(self,name):
        self.name = name
        self.cards = []
        self.reaction_mean = 0
        self.turn = True


    def add_to_cards(self,add_cards):
        self.cards.extend(add_cards)

    def get_time_score(self, time):
        return scipy.stats.norm(self.reaction_mean, self.reaction_var).cdf(time)

    def remove_top_card(self):
        card=self.cards.pop(0)
        return card

class game():
    def __init__(self,num_players):
        self.card_pass = "cat"
        self.players_playing = []
        self.cards_on_table = []
        self.players_turn = 'cat'
        self.player_lst = []
        self.face_bool = False
        self.card_down_before_slap = 'cat'
        self.someone_slapped = "cat"
        self.attempts_equal_bound = False
        self.attempt_bound = "cat"
        self.num_attempts = "cat"
        self.player_who_placed_face_card = "cat"

    def player_no_cards(self, player_num):
        self.players_playing.remove(player_num)

    def add_to_card_on_table(self,card):
        self.cards_on_table.append(card)

    def player_gets_cards_on_table(self, player):
        player.add_to_cards(self.cards_on_table)
        self.cards_on_table = []
        print(player.name, 'got the deck!')

    def check_slappable(self, top_card):
        slappable = False
        if len(self.cards_on_table) == 1:
            return False
        # print('check here', self.cards_on_table)
        if self.cards_on_table[-2][:-1] == top_card[:-1]:          #two in a row
            slappable = True
            # print('Slappable Two in a row')
            # print('topcard', top_card[0])
            # print('match card', self.cards_on_table[-2][0])

        if len(self.cards_on_table) >= 3:                       #sandwich
            if self.cards_on_table[-3][:-1] == top_card[:-1]:
                slappable = True
                # print('Slappable Sandwich')

        if top_card[:-1] not in ['1', '13']:          #consecutives
            diff = int(self.cards_on_table[-2][:-1]) - int(top_card[:-1])
            # print('diff', diff)
            if diff == 1 or diff == -1:
                slappable = True
                # print('Slappable Consecutive')
        if top_card[:-1] == '13':
            if self.cards_on_table[-2][:-1] == '1' or self.cards_on_table[-2][:-1]== '12':
                slappable = True
                # print('Slappable Consecutive')
        if top_card[:-1] == '1':
            if self.cards_on_table[-2][:-1]== '2' or self.cards_on_table[-2][:-1]== '13':
                slappable = True
                # print('Slappable Consecutive')

        return slappable

    def card_down_before_slap_opportunity(self,event):
        same_player = 'cat'
        if self.face_bool:
            ###deal with placing cards down before slappable
            if (event.key == pygame.K_k and player.name == "Jeff") or (
                    event.key == pygame.K_a and player.name == 'Peter'):
                # returnhereindo
                self.card_down_before_slap = True
                same_player = True
                print('card down bro ')

        if not self.face_bool:
            if (event.key == pygame.K_a and player.name == "Peter") or (
                    event.key == pygame.K_k and player.name == 'Jeff'):
                # returnhereindo
                self.card_down_before_slap = True
                print('card down bro 2')
                same_player = True
        return same_player
    def detect_who_slapped(self,event):
        player_slap = 'cat'
        if event.key == pygame.K_z:
            self.someone_slapped = True
            player_slap = player_lst[0]
        elif event.key == pygame.K_m:
            self.someone_slapped = True
            player_slap = player_lst[1]
        return player_slap

    def burn(self,player):
        card = player.remove_top_card()
        if len(self.cards_on_table) <=2:
            self.cards_on_table.insert(0,card)
        for i in range(len(self.cards_on_table)):
            display_card(self.cards_on_table[i], 140 + 60 * i, 200)

    def update_card_counter(self):
        player1_card_text = font.render('Player 1 Cards:' + " " + str(len(self.player_lst[0].cards)), False,
                                        (0, 0, 0))
        player2_card_text = font.render('Player 2 Cards:' + " " + str(len(self.player_lst[1].cards)), False,
                                        (0, 0, 0))
        pygame.draw.rect(WIN, (55, 159, 90), clear_rect_card_count)
        WIN.blit(player1_card_text, (50, 400))
        WIN.blit(player2_card_text, (250, 400))
        pygame.display.update()
    def update_player_text(self,play_idx):
        player_playing_text = font.render('Playing: Player ' +str(play_idx + 1), False,
                                        (0, 0, 0))
        pygame.draw.rect(WIN, (55, 159, 90), clear_rect_playing_txt)
        WIN.blit(player_playing_text, (150, 100))
        # (50, 100)
        pygame.display.update()

    def discrete_click(self,player):
        end = False
        print('enter here', player.name)
        while not end:
            down = False
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a and player.name == "Jeff":
                        play = True
                        down = True
                        print('pressed a')
                    elif event.key == pygame.K_k and player.name == 'Peter':
                        play = True
                        down = True
                        print('pressed k')

                    while down:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYUP:
                                print('keyup')
                                return play
    def update_graphics_table(self):
        top_3_cards_on_table = self.cards_on_table[-3:]
        for i in range(len(top_3_cards_on_table)):  # display cards on table
            display_card(top_3_cards_on_table[i], 140 + 60 * i, 200)

    def face_card_ends(self):
        self.player_gets_cards_on_table(self.player_who_placed_face_card)
        self.update_card_counter()
        self.num_attempts = 0
        # clearcards
        WIN.fill((55, 159, 90))
        pygame.display.update()
        self.attempt_bound = -1
        self.face_bool = False
        self.attempts_equal_bound = False
        self.player_who_placed_face_card = "cat"
    # slappable = False
# if Deck[-1][0] == top_card[0]:
#     slappable = True
#
# if Deck[-2][0] == top_card[0]:
#     slappable = True
#
# if top_card[0] not in ['A', 'J', 'Q', 'K']:
#     diff = int(Deck[-1][0]) - int(top_card[0])
#     if diff == 1 or diff ==-1:
#         slappable = True
#
#

sample_lst = []

#get the reaction times x number of times
start = time.time()
end = time.time()

reaction_time = end-start
sample_lst.append(reaction_time)

mean = np.average(sample_lst)
var = np.var(sample_lst)


player2 = player('Peter')
player1 = player('Jeff')
player_lst = [player1, player2]
while Deck != []:
    for player in player_lst:
        card = random.choice(Deck)
        player.add_to_cards([card])
        Deck.remove(card)






WIDTH, HEIGHT = 500,500
WIN = pygame.display.set_mode((WIDTH,HEIGHT))
folder_path = "C:/Users/peter/Downloads/pcp/Spade"
file_name = "card_4_spade.png"
image = pygame.image.load(os.path.join(folder_path,file_name))
Suit_dict = {'S': "Spade", "H": "Heart", "D": "Diamond", "C": "Clover"}
Drawing_dict = {"1": 4, "11": 1, "12": 2, "13":3}
clear_rect_card_count = pygame.Rect(50, 400, 400, 50)  # Change coordinates and size as needed
clear_rect_playing_txt = pygame.Rect(50, 100, 400, 50)
# WIN.blit(player1_card_text, (50, 400))
# WIN.blit(player2_card_text, (250, 400))

def display_card(card, x,y):

    suit = Suit_dict[card[-1]]
    card_num = card[:-1]
    folder_path = "C:/Users/peter/Downloads/pcp/" + suit
    file_name = 'card_' + card_num + "_" + suit.lower() + ".png"
    image = pygame.image.load(os.path.join(folder_path, file_name))
    WIN.blit(image,(x,y))
    pygame.display.update()


clock = pygame.time.Clock()
FPS = 120
def main(num_players):
    run =True
    clock = pygame.time.Clock()
    ERS= game(num_players)
    play_idx = 0
    ERS.attempt_bound = -1
    ERS.num_attempts= 0

    ERS.player_lst = [player1, player2]                     #temp
    ERS.players_playing = [player1, player2]


    WIN.fill((55, 159, 90))                                             #intialize background green
    pygame.display.update()


    pygame.font.init()  # you have to call this at the start,               #font stuff
    # if you want to use this module.



    while run:
        print('play idx cuh2', play_idx)
        print('-------------------------------NEW WHILE LOOP ITERATIONS-------------------------------')
        print('play idx', play_idx % num_players)
        # print('ERS.num_attempts', ERS.num_attempts)
        # print('attempt bound', ERS.attempt_bound)
        ERS.update_card_counter()
        ERS.card_down_before_slap = False
        clock.tick(FPS)
        player = ERS.player_lst[play_idx % num_players]            #get the player who is playing by using a moving index that goes thorugh the list of player objects

        if player not in ERS.players_playing:                               #If the player selected is no longer playing then just go to the next person
            play_idx +=1
        else:
            for event in pygame.event.get():                                                #quitting X
                if event.type == pygame.QUIT:
                    run = False
            play = ERS.discrete_click(player)               #problem here

            if play:       #deletediscrete
                print('player playing', player.name, '-----------------------------------')
                ERS.update_player_text(play_idx  %num_players)

                top_card = player.cards[0]
                print('top card first!', top_card)

                ERS.add_to_card_on_table(top_card)                              #player puts card onto table
                player.remove_top_card()                                #abstract
                ERS.update_card_counter()
                if ERS.attempt_bound > 0:                                            #increment num attempts only if it has been previously set to positive
                    ERS.num_attempts +=1

                if player.cards == []:                                   #remove player if their hand is empty
                    ERS.player_no_cards(player)
                # display cards on table
                ERS.update_graphics_table()
                # display cards on table
                if top_card[:-1] in ['1', '11', '12', '13']:
                    ERS.attempt_bound = Drawing_dict[top_card[:-1]]
                    ERS.num_attempts = 0
                    ERS.face_bool = True
                    ERS.player_who_placed_face_card = player
            #returnsol
                    ########NOTES

                    ###Each player takes a turn
                    ###before the turn, initialize the number of repeats to 0 and increment by 1 each time they click
                    ###if they do not get a face card, then their turn is over
                    ##else: turn is not over and increment the turn by 1
                    ### when the ERS.num_attempts is reached, the turn is over and if no one slapped, then the palyers gets all the cards and plays again
                    ###throughout all this, the option to slap is open every single time a card is placed down
                    #need to implement the damn turns and skip list

                slappable = False
                ERS.someone_slapped = False

                print('went here')
                if ERS.check_slappable(top_card):
                    slappable = True

                if ERS.num_attempts == ERS.attempt_bound:  # notinstaget
                    ERS.attempts_equal_bound = True

                if ERS.attempts_equal_bound:
                    if not slappable:
                        run = True
                        while run:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    run = False
                                    break
                                elif event.type == pygame.KEYDOWN:
                                    if  (event.key == pygame.K_k and ERS.player_who_placed_face_card.name == "Peter") \
                                            or (event.key == pygame.K_a and ERS.player_who_placed_face_card.name == 'Jeff'):  # notinstaget
                                        ERS.face_card_ends()
                                        break

                    #if not slappable, simply hve the player pick up his stuff
                        #player who placed face card must play next turn key in order to pick up
                    # if it is slappable defer the stuff to slappable portion


                if slappable:                       #slappable stuff
                    player_slap = "cat"

                    while not ERS.someone_slapped:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                run = False
                                break
                            elif event.type == pygame.KEYDOWN:
                                if not ERS.attempts_equal_bound:
                                    same_player = ERS.card_down_before_slap_opportunity(event)  # card goes down before slap
                                #returnnow
                                if ERS.card_down_before_slap:
                                    break

                                player_slap = ERS.detect_who_slapped(event)             #player slaps
                                if player_slap != "cat":
                                    ERS.player_gets_cards_on_table(player_slap)
                                    play_idx = ERS.player_lst.index(player_slap)
                                    ERS.update_card_counter()
                                    ERS.attempt_bound = -1
                                    if ERS.someone_slapped:
                                        break

                                if ERS.num_attempts == ERS.attempt_bound and (event.key == pygame.K_k and player.name == "Peter") or (
                                                        event.key == pygame.K_a and player.name == 'Jeff'):  # notinstaget
                                    ERS.someone_slapped = True
                                    ERS.face_card_ends()
                                    break



                        if ERS.card_down_before_slap:                               #jump out of all stuff when player keeps playing
                            player = ERS.player_lst[(play_idx +1) % num_players]
                            ERS.update_player_text((play_idx +1) % num_players)
                            #downbeforereturn
                            print(' PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM ')
                            top_card = player.cards[0]
                            print('top_card new', top_card)
                            ERS.someone_slapped = True
                            ERS.add_to_card_on_table(top_card)  # player puts card onto table
                            player.remove_top_card()
                            ERS.update_card_counter()
                            ERS.update_graphics_table()
                            print('play idx cuh', play_idx)
                    if not ERS.card_down_before_slap:
                        print('exit while loop and clear')
                        WIN.fill((55, 159, 90))
                        pygame.display.update()
                        ERS.num_attempts =0
                        ERS.face_bool = False
                if not slappable or ERS.card_down_before_slap:

                    #######burn cards
                    burn = False

                    #if not slap, then just go to next player!

                    if not ERS.card_down_before_slap:
                        if not ERS.face_bool:
                            # returnnow
                            play_idx +=1
                        if ERS.face_bool and ERS.num_attempts == 0:
                            print('Nikki Haley')
                            play_idx +=1
                    # if ERS.card_down_before_slap:
                    #     print('---------------------NIKKI HALEY--------------------------')
                    #     play_idx += 1
                    if ERS.num_attempts == ERS.attempt_bound:                   #notinstaget
                        animal = "dog"
                        while animal != "cat":
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    animal = "cat"
                                    break
                                elif event.type == pygame.KEYDOWN:
                                    ###petershih
                                    if (event.key == pygame.K_k and player.name == "Peter") or (
                                            event.key == pygame.K_a and player.name == 'Jeff'):
                                        animal = "cat"
                                    break


                        ERS.face_card_ends()

                                   #cleartable if facecard terminates
                        #returnindo


            # print(player1.name, len(player1.cards))
            # print(player2.name, len(player2.cards))
        for player in ERS.player_lst:
            if len(player.cards) == 52:
                WIN.fill((55, 159, 90))
                player_playing_text = font.render(player.name + 'has Won!', False,
                                                  (0, 0, 0))
                pygame.draw.rect(WIN, (55, 159, 90), clear_rect_playing_txt)
                WIN.blit(player_playing_text, (150, 100))
                # (50, 100)
                pygame.display.update()

    pygame.quit()




main(2)

### press space to flip card
#if it is not a face card then it is the other players turn
