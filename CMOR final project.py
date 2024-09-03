import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class U_container():
    def __init__(self):
        self.lst = []

    def add(self,stuff):
        self.lst.append(stuff)

    def get_dict(self):
        return self.lst

class counter():
    def __init__(self):
        self.dict = defaultdict(lambda:0)

    def add_order_lst(self,player):
        self.dict[player] +=1

    def get_dict(self):
        return self.dict

class go_fish():

    def __init__(self, num_players,verbose,order_lst):
        self.knowledge_have = []
        self.knowledge_not_have = []
        self.full_sets = []
        self.card_container = []
        self.order_lst = order_lst
        self.turn = True
        self.deck = []
        self.num_players = num_players
        self.verbose = verbose
    def remove_from_order_lst(self,i):
        if self.verbose:
            print('player', i, "was removed")
        self.order_lst.remove(i)
    def turn_false(self):
        self.turn = False
    def turn_true(self):
        self.turn = True

    def take_card_knowledge_update(self,ask_card, i,j):
        # if self.verbose:
        #     print('update problem delete')
        #     print('ask card delete', ask_card)
        #     print('i ', i)
        #     print('j', j )
        #     print('knowledge have j', self.knowledge_have[j])

        if ask_card in self.knowledge_have[j]:
            self.knowledge_have[j].pop(ask_card)
        if ask_card in self.knowledge_not_have[i]:
            self.knowledge_not_have[i].remove(ask_card)
        # if self.verbose:
        #     print('js hand', self.card_container[j])
        for card in self.card_container[j]:
            if card == ask_card:
                self.knowledge_have[i][ask_card] +=1
        #this case only happens when you ask for a card that j does not have
        if self.knowledge_have[i][ask_card] == 0:
            self.knowledge_have[i][ask_card] =1

        self.knowledge_not_have[j].add(ask_card)

    def go_fish(self, ask_card,i,j):
        # if self.verbose:
        #     print('deck', self.deck)
        go_fish_card = random.choice(self.deck)
        if self.verbose:
            print('player', i, "fished card:", go_fish_card)
        # if self.verbose:
        #     print('go fish verify')
        #     print('view deck', self.deck)
        #     print('go fish card', go_fish_card)
        self.deck.remove(go_fish_card)
        self.card_container[i].append(go_fish_card)
        self.turn = False
        return go_fish_card

    def remove_dealt_four_card(self,card_star,i):
        for z in self.order_lst:
            self.knowledge_not_have[z].add(card_star)
            if card_star in self.knowledge_have[z]:
                self.knowledge_have[z].pop(card_star)

        self.card_container[i] = [j for j in self.card_container[i] if j != card_star]
        # if self.verbose:
        #     print('remove_dealt four_card')
        #     print('should have cleared',card_star )
        #     print("knowledge_have", self.knowledge_have)
        self.full_sets[i].append(card_star)

    def take_card(self, ask_card, i, j,card_count_dict_actual):
        if self.verbose:
            print('player', i, 'takes', ask_card,  'from player', j)
        len_j_before = len(self.card_container[j])
        #knoledge comeback
        self.card_container[j] = [i for i in self.card_container[j] if i != ask_card]
        len_j_after = len(self.card_container[j])
        for _ in range(len_j_before - len_j_after):
            self.card_container[i].append(ask_card)
            card_count_dict_actual[ask_card] +=1
        # if self.verbose:
        #     print('num occurents', len_j_before - len_j_after)
        #     print('take card verify')

        return card_count_dict_actual

    def produce_places(self,dog):
        order_lst = []
        skip_lst = []

        global_max_val = 0

        while len(skip_lst) != len(dog):
            max_val = -1
            idx = None
            for i, player in enumerate(dog):  # Using enumerate to get both index and player
                if i not in skip_lst:
                    if len(player) > max_val:
                        max_val = len(player)
                        idx = i

            if max_val > global_max_val:
                global_max_val = max_val
            if idx is not None:
                order_lst.append(idx)
                skip_lst.append(idx)

        ties = []
        for num in range(len(dog)):
            if len(dog[num]) == global_max_val:
                ties.append(num)

        random.shuffle(ties)
        cut_order_lst = order_lst[len(ties):]
        ties.extend(cut_order_lst)
        order_lst = ties
        return order_lst

    def check_four_groups(self,ask_card,i,j, card_holder_dict, card_count_dict_actual):
        if card_count_dict_actual[ask_card] == 4:
            self.card_container[i] = [j for j in self.card_container[i] if j != ask_card]

            #discard the four set from knowledge
            if ask_card in self.knowledge_have[i]:
                self.knowledge_have[i].pop(ask_card)

            for j in self.order_lst:
                self.knowledge_not_have[j].add(ask_card)
            self.full_sets[i].append(ask_card)

    def clear_guaranteed_four_sets(self, four_card_lst, i,card_holder_dict):
        if self.verbose:
            print('guaranteed four sets removal')
        for ask_card in four_card_lst:
            # adjust knowledge (not including the discarding)
            for j in self.order_lst:
                if j!=i:
                    self.take_card_knowledge_update(ask_card, i, j)

            for j in card_holder_dict[ask_card]:
                if j!=i:
                    if self.verbose:
                        print('player', i, 'takes card' , ask_card, 'from player', j)
                    # remove cards from players hand, player doesn't
                    self.card_container[j] = [i for i in self.card_container[j] if i != ask_card]
            self.card_container[i] = [j for j in self.card_container[i] if j != ask_card]  # remove cards from i's hand, and remove it from the knowledge, update full set

            # discard the four set from knowledge
            if ask_card in self.knowledge_have[i]:
                self.knowledge_have[i].pop(ask_card)
            for j in self.order_lst:
                self.knowledge_not_have[j].add(ask_card)
            # if self.verbose:
            #     print('clear_guaranteed')
            #     print('should have cleared', ask_card)
            #     print("knowledge_have", self.knowledge_have)
            self.full_sets[i].append(ask_card)


    def initialize(self, num_players):
        for i in range(num_players):
            self.card_container.append([])
        Ranks = ['A', '2', '3', '4',
                 '5', '6', '7', '8',
                 '9', '10', 'J', 'Q', 'K']
        for j in range(4):
            for rank in Ranks:
                self.deck.append(rank)
        # initialize knowledge to size of players
        for i in range(num_players):
            self.knowledge_have.append(defaultdict(lambda: 0))
            self.knowledge_not_have.append(set())
            self.full_sets.append([])
        for j in range(3):
            for i in self.order_lst:
                card = random.choice(self.deck)
                self.deck.remove(card)
                self.card_container[i].append(card)

    def mod_fullsets(self,card,i):
        self.full_sets[i].append(card)
    def get_knowledge_have(self):
        return self.knowledge_have
    def set_order_lst(self,lst):
        self.order_lst= lst
    def get_knowledge_not_have(self):
        return self.knowledge_not_have

    def get_full_sets(self):
        return self.full_sets

    def get_card_container(self):
        return self.card_container

    def get_order_lst(self):
        return self.order_lst

    def get_turn(self):
        return self.turn

    def get_deck(self):
        return self.deck

    def all_dealt_four_groups(self):
        self.card_container = [['2','2','2','2','8'],['3','3','3','3','9'], ['4','4','4','4','10'], ['5','5','5','5','J'], ["7","7","7","7","Q"]]
        self.deck = ['A', '6', 'K', 'A', '6', '8', '9', '10', 'J', 'Q', 'K', 'A', '6', '8', '9', '10', 'J', 'Q', 'K', 'A', '6', '8',
         '9', '10', 'J', 'Q', 'K']

    def block(self,game_info,order_lst_mod, i,j,card_count_dict_actual,card_holder_dict,ask_card):
        if self.verbose:
            print('player', i, 'asks player', j, 'for card', ask_card)
        #no matter if they go fish or not, the same knowledge is obtained
        self.take_card_knowledge_update(ask_card,i,j)
        if ask_card in game_info.get_card_container()[j]:
            # if self.verbose:
            #     knowledge_have_print = []
            #     for know in game_info.get_knowledge_have():
            #         knowledge_have_print.append(dict(know))
            #     print('knowledge have i after', knowledge_have_print)
            #     # part b
            card_count_dict_actual = game_info.take_card(ask_card, i, j, card_count_dict_actual)
            game_info.check_four_groups(ask_card, i,j, card_holder_dict, card_count_dict_actual)
        else:
            if game_info.get_deck() != []:
                go_fish_card = game_info.go_fish(ask_card, i, j)
                card_count_dict_actual[go_fish_card] += 1
                game_info.check_four_groups(go_fish_card, i,j, card_holder_dict, card_count_dict_actual)
            game_info.turn_false()

def run_sim_helper(num_players, memory_lst, memory_lst_not, order_lst,verbose):
    game_info = go_fish(num_players,verbose,order_lst)
    game_info.initialize(num_players)
    card_container = "cat"
    # print('shuffled order lst? remove', game_info.get_order_lst())
    end_condition = []
    for i in range(num_players):
        end_condition.append([])
    cond_iter= 0
    while card_container != end_condition:
        cond_iter +=1
        for i in game_info.get_order_lst():
            if verbose:
                print("----------------------------------------player",i,"----------------------------------------")
            order_lst_mod = game_info.get_order_lst().copy()
            order_lst_mod.remove(i)
            game_info.turn_true()
            player_turn = 1
            while game_info.get_turn():
                if verbose:
                    print("*****************player", i, '','turn',player_turn,"*****************")
                    print('card container before player plays', game_info.get_card_container())
                    print('knowledge_have before player plays', (game_info.get_knowledge_have()))

                dealt_four_match = False
                if cond_iter == 1 and player_turn ==1:
                    dd = defaultdict(lambda: 0)
                    for card in game_info.get_card_container()[i]:
                        dd[card] += 1
                        if dd[card] == 4:
                            dealt_four_match = True
                            card_star =card
                            break
                    if dealt_four_match:
                        print('player',i, ' was','dealt four match')
                        game_info.remove_dealt_four_card(card_star,i)
                    ########special case dealt four match

                if not dealt_four_match:
                    #this is the case where the last card need to complete the game is in the pond
                    global ask_card
                    ask_card = "cat"
                    ###initialize the data structures to find which cards can give a full set and the players to ask for
                    card_count_dict = defaultdict(lambda: 0)
                    card_holder_dict = defaultdict(lambda:[])
                    memory_lvl = memory_lst[i]
                    memory_lvl_not = memory_lst_not[i]
                    global found_bool
                    found_bool = False
                    i_cards = game_info.get_card_container()[i]
                    for card in i_cards:
                        card_count_dict[card] +=1
                                                                                ###gather info on what people have
                    card_count_dict_actual = card_count_dict.copy()

                    #update knowledge of peoples hands
                    for card in np.unique(i_cards):
                        for j in range(len(game_info.get_order_lst())):
                                if j!=i:
                                    if card in game_info.get_knowledge_have()[j]:
                                        found_bool = True
                                        card_count_dict[card]+= game_info.get_knowledge_have()[j][card]

                    #update which people hold which cards
                    for card in np.unique(i_cards):
                        for j in range(len(game_info.get_order_lst())):
                            if card in game_info.get_card_container()[j]:
                                card_holder_dict[card].append(j)

                    if len(game_info.get_order_lst()) == 1:
                        if verbose:
                            print("game end deck nonempty")
                            print('card container', game_info.get_card_container() )
                            print('deck', game_info.get_deck())
                        game_info.turn_false()
                        game_info.remove_from_order_lst(i)
                        card_container = end_condition
                        for card in game_info.get_deck():
                            game_info.mod_fullsets(card, i)
                        game_info.set_order_lst([])
                        break

                    U = random.random()
                    if found_bool:
                        if U > memory_lvl:
                            if verbose:
                                print('found bool random strategy')
                            ask_card = random.choice(i_cards)
                            j = random.choice(order_lst_mod)
                            game_info.block(game_info,order_lst_mod, i,j,card_count_dict_actual,card_holder_dict,ask_card) # comeback breaks
                        if U < memory_lvl:
                            if verbose:
                                print('found bool strategy')
                            max_val = max(card_count_dict.values())
                            if max_val != 4:
                                found_bool = False
                            if max_val == 4:
                                four_card_lst = []
                                for card in card_count_dict:
                                    if card_count_dict[card] == 4:
                                        four_card_lst.append(card)
                                game_info.clear_guaranteed_four_sets(four_card_lst,i,card_holder_dict)


                    if found_bool == False:
                        if U > memory_lvl_not:
                            if verbose:
                                print('not foundbool random strategy')

                            ask_card = random.choice(i_cards)
                            j = random.choice(order_lst_mod)
                            game_info.block(game_info,order_lst_mod, i,j,card_count_dict_actual,card_holder_dict,ask_card)
                        ####################################################
                        if U < memory_lvl_not:
                            if verbose:
                                print('not foundbool strategy')
                            if len(game_info.get_knowledge_have()[i]) != 0:
                                ask_card = random.choice(list(game_info.get_knowledge_have()[i].keys()))
                            else:
                                ask_card = random.choice(i_cards)
                            found_not_bool = False
                            # for player in order_lst_mod:
                            #     if ask_card in game_info.get_knowledge_not_have()[player]:
                            #         j = player
                            #         # print('go fish guaranteed delete')
                            #         found_not_bool = True
                            #         break
                            #indoremove
                            if verbose:
                                if found_not_bool:
                                    print('found not bool')
                            if not(found_not_bool and game_info.get_deck()!=[]):
                                # print('deck size', len(game_info.get_deck()))
                                # print('found_not bool', found_not_bool)
                                # print('does this happen a lot? delete')
                                j = random.choice(order_lst_mod)
                                ###go fish

                            game_info.block(game_info, order_lst_mod, i, j, card_count_dict_actual, card_holder_dict,ask_card)

                if game_info.get_card_container()[i] == []:
                    game_info.remove_from_order_lst(i)
                    game_info.turn_false()

                for z in order_lst_mod.copy():
                    if game_info.get_card_container()[z] == []:
                        game_info.remove_from_order_lst(z)
                        order_lst_mod.remove(z)
                player_turn+=1
                if verbose:
                    print('card container before player plays', game_info.get_card_container())
                    print('knowledge_have after player plays', game_info.get_knowledge_have())

                if game_info.get_card_container() == end_condition:
                    card_container= game_info.get_card_container()
                    break
            if game_info.get_card_container() == end_condition:
                card_container = game_info.get_card_container()
                break
    #unremove
    # print('remove indo', game_info.produce_places(game_info.get_full_sets()))
    if verbose:
        print('-----------------------------------------')
        print('result', game_info.get_full_sets())
        print('result', game_info.produce_places(game_info.get_full_sets()))
        print('-----------------------------------------')
    return game_info.produce_places(game_info.get_full_sets())

def sim_perc_win(num_players,trials,have_mem, not_have_mem,order_lst,verbose,shuffle):
    places_player = []
    player_places = []
    for i in range(num_players):
        player_places.append([])
        places_player.append([])
    for z in range(trials):
        #unremove
        if verbose:
            print("************************************************************ITERATION", z, "*********************************************")
        if shuffle:
            random.shuffle(order_lst)
        #unremove
        # print('shuffle remove', order_lst)
        count.add_order_lst(order_lst[0])
        single_sim = run_sim_helper(num_players, have_mem, not_have_mem, order_lst.copy(),verbose)
        # print(single_sim)
        d = {}
        #d keys are player number and the value is the place that they got
        for i in range(num_players):
            d[single_sim[i]] = i

        #player_places is a dict with keys as place and values as list of players that got that place
        #places_player is a dict with keys as player and values as list of places that player got

        for place in range(num_players):
            player_places[place].append(single_sim[place])
        for player in range(len(single_sim)):
            places_player[player].append(d[player])
        # if verbose:
        #     print('result', single_sim)

    player_places_count = {}
    for place in range(len(player_places)):
        count_dict = defaultdict(lambda:0)
        for player in player_places[place]:
            count_dict[player] +=1 / trials
        player_places_count[place] = dict(count_dict)

    places_player_count = {}
    for player in range(len(places_player)):
        count_dict = defaultdict(lambda: 0)
        for places in places_player[player]:
            count_dict[places] += 1 / trials
        places_player_count[player] = dict(count_dict)

    return dict(places_player_count), dict(player_places_count)

def run_sim(num_players,trials,have_mem, not_have_mem,order_lst,verbose,trials2,shuffle):
    trials_result_dict = defaultdict(lambda:[])
    mean_dict = {}
    var_dict = {}
    for i in range(trials2):
        print('variance trial', i)
        dict1, dict2 = sim_perc_win(num_players,trials,have_mem, not_have_mem,order_lst,verbose,shuffle)
        first_place_dict = dict2[0]
        for player in first_place_dict:
            trials_result_dict[player].append(first_place_dict[player])
    confidence_interval_dict = {}
    for player in trials_result_dict:
        perc_2_5 = np.percentile(trials_result_dict[player], 2.5)
        perc_97_5 = np.percentile(trials_result_dict[player], 97.5)
        mean_dict[player]=(np.average(trials_result_dict[player]))
        var_dict[player]=(np.var(trials_result_dict[player]))
        confidence_interval_dict[player] = (perc_2_5, perc_97_5)
    return mean_dict, var_dict, confidence_interval_dict

#############################parameters for simulation!##############################
num_players = 5

all_ones = [1 for i in range(num_players)]
all_zeros = [0 for i in range(num_players)]
order_lst = [i for i in range(num_players)]
last_1 = [0 for i in range(num_players-1)]
last_1.append(1)
first_1 = [1]
first_1.extend([0 for i in range(num_players-1)])

print('bruh', first_1)
half_all = [0.5 for i in range(num_players)]
memory_test = [0,0.25,0.5,0.75,1]
# result = sim_perc_win(num_players,100,last_1,last_1,order_lst,False)

count = counter()

#simulation are ran here
#note that 6th parameter is the verbose parameter. Set to True to get print statements to follow the game as it is played
#set parameter to false for faster simulation run time
result = run_sim(num_players,10,first_1,first_1,order_lst,True,1,True)

print('dict', dict(count.get_dict()))
print('mean')
print(result[0])
print('variance')
print(result[1])
print('CI dict')
print(result[2])


def plot_line_chart(x_lst, y_lst):
    plt.figure(figsize=(8, 6))  # Set the size of the figure

    plt.plot(x_lst, y_lst, marker='o', linestyle='-')  # Plotting the line chart with markers at data points
    plt.title('Line Chart')  # Set the title of the plot
    plt.xlabel('X-axis')  # Label for the x-axis
    plt.ylabel('Y-axis')  # Label for the y-axis
    plt.grid(False)  # Remove gridlines

    plt.show()  # Display the plot



# Plotting the line chart with the example data
def weird_sim():
    x_lst = []
    y_lst = []
    for num_players in range(2,17):
        print('num_player', num_players)
        first_1 = [1]
        first_1.extend([0 for i in range(num_players - 1)])
        all_zeros = [0 for i in range(num_players)]
        order_lst = [i for i in range(num_players)]

        result = run_sim(num_players, 100, first_1, all_zeros, order_lst, False, 5, True)

        x_lst.append(num_players)
        avg_lst = []
        for i in range(1,num_players):
            avg_lst.append(result[0][i])

        y_lst.append(result[0][0]/np.average(avg_lst))
    plot_line_chart(x_lst, y_lst)
# weird_sim()
# # Your dictionary with player indices as keys and win percentages as values
# win_percentages = result[0]
# # Extracting player indices and win percentages for plotting
# players = list(win_percentages.keys())
# percentages = list(win_percentages.values())
#
# # Plotting the bar graph with light blue color
# plt.figure(figsize=(8, 6))
# plt.bar(players, percentages, color='lightblue')  # Setting the color to light blue
# plt.xlabel('Player')
# plt.ylabel('Win Percentage')
# plt.title('Expected Win %: All Players Play Randomly, No Shuffle')
# plt.xticks(players)  # Setting the x-ticks as player indices
# plt.tight_layout()
#
# # Show the plot
# plt.show()


# # # Given dictionary
# # data = {1: (0.11474999999999999, 0.26000000000000006), 2: (0.11474999999999999, 0.2700000000000001), 4: (0.10999999999999999, 0.26000000000000006), 3: (0.11474999999999999, 0.25000000000000006), 0: (0.16475, 0.35000000000000014)}
# #
# data = result[2]
# # Extracting player numbers and confidence intervals
# players = list(data.keys())
# intervals = [data[player] for player in players]
# lower_bounds = [interval[0] for interval in intervals]
# upper_bounds = [interval[1] for interval in intervals]
#
# # Calculating means
# means = [(upper + lower) / 2 for upper, lower in zip(upper_bounds, lower_bounds)]
#
# # Plotting
# plt.figure(figsize=(8, 6))
#
# # Plotting mean points
# plt.plot(players, means, 'o', color='black', label='Mean')
#
# # Plotting lines from mean to upper and lower bounds
# for player, mean, upper, lower in zip(players, means, upper_bounds, lower_bounds):
#     plt.plot([player, player], [mean, upper], color='blue')  # Line to upper bound
#     plt.plot([player, player], [mean, lower], color='blue')  # Line to lower bound
#
# plt.xlabel('Player')
# plt.ylabel('Win % Confidence Interval')
# plt.title('Win % 95% Confidence Interval for Players')
# plt.legend(['Mean'])
# plt.xticks(players)  # Setting player numbers on x-axis
# plt.show()
#
#
#
