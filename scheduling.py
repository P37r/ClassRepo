# import gurobipy as gp
# from gurobipy import GRB
#
# def main(n, lst):
#     # Create a Gurobi model
#     model = gp.Model()
#
#     # Create the x_{i,j} variables if job i gets placed in position j
#     x = {}
#     for i in range(1, n + 1):
#         for j in range(1, n + 1):
#             x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
#
#
#     # Add constraints: sum from i=1 to n (sum from j=1 to n (x_{i,j})) = 1
#     for i in range(1, n + 1):
#         model.addConstr(gp.quicksum(x[(i, j)] for j in range(1, n + 1)) == 1)
#
#
#     for j in range(1, n + 1):
#         model.addConstr(gp.quicksum(x[(i, j)] for i in range(1, n + 1)) == 1)
#
#     # Define variables w_p
#     w_p = {}
#     for p in range(1, n + 1):
#         w_p[p] = model.addVar(vtype=GRB.CONTINUOUS, name=f'w_{p}')
#
#     # Add constraints for w_p
#     model.addConstr(w_p[1] == 0)
#     model.addConstrs(
#         (w_p[p] == gp.quicksum(gp.quicksum(x[(i, j)] * lst[i - 1] for j in range(1, p)) for i in range(1, n + 1))
#          for p in range(2, p)))
#
#
#     # Set the objective function: minimize sum from p=1 to n (w_p)
#     model.setObjective(gp.quicksum(w_p[p] for p in range(1, n + 1))/n, GRB.MINIMIZE)
#
#     # Optimize the model
#     model.optimize()
#
#     # Print the solution
#     if model.status == GRB.OPTIMAL:
#         # print("Optimal Solution:")
#         wait_lst = []
#         for p in range(1, n + 1):
#             wait_lst.append(w_p[p].X)
#         dict = {}
#         for i in range(1, n + 1):
#             for j in range(1, n + 1):
#                 if x[(i,j)].X == 1:
#                     dict[i] = j
#
#         return dict, model.ObjVal,wait_lst
#     else:
#         print("No solution found")
#
#
# times = [222,25,2,2,2,2,34,6,2,23,23,1,2,3,32,111,100]
#
# opt_res = main(len(times),times)
# res_dict = opt_res[0]
#
# print('dict',res_dict)
# print('Obj Val',opt_res[1])
#
# wait_lst = opt_res[2]
#
# wait_dict = {}
# for i in res_dict:
#     wait_dict[i] = wait_lst[res_dict[i]-1]
# print('wait',wait_dict)
#
#
#
#
# time_sort = times.copy()
# time_sort.sort()
#
# def calc(lst):
#     wait_time = {}
#     for i in range(len(lst)):
#             sub_lst = lst[:i]
#             val = sum(sub_lst)
#             wait_time[i+1] = val
#     return wait_time
#
# res = calc(times)
# res_sorted =calc(time_sort)
#
# def difference(x,y):
#     res_dict = {}
#     for i in x:
#         res_dict[i] = x[i] - y[i]
#
#     return res_dict
#
#
# print('difference',difference(wait_dict,res))
#
# print('unsorted',sum(res.values())/len(res))
# print('heuristic', sum(res_sorted.values())/len(res_sorted))
# print('optimal', opt_res[1])


lst = [4, 2, 7, 3, 5, 6, 6, 8, 2, 4, 6, 3, 7, 1, 5, 8]


dict = {1: "50-60", 2: "70", 3: "80", 4: "90", 5: "2000", 6: "2010", 7: "2010's rap", 8: "2020"}

lst_mod = []
for i in lst:
    lst_mod.append(dict[i])
print(lst_mod)



_________


def run_sim(num_players, memory_lst, memory_lst_not, order_lst):
    # a list of all the ranks
    Ranks = ['A', '2', '3', '4',
             '5', '6', '7', '8',
             '9', '10', 'J', 'Q', 'K']
    deck = []
    for j in range(4):
        for rank in Ranks:
            deck.append(rank)
    #initialize knowledge to size of players
    knowledge_have = []
    knowledge_not_have = []
    full_sets = []
    for i in range(num_players):
        knowledge_have.append(set())
        knowledge_not_have.append(set())
        full_sets.append([])


    card_container = []

    for i in range(num_players):
        initial_cards = random.sample(deck,5)
        for card in initial_cards:
            deck.remove(card)
        card_container.append(initial_cards)


    #Play game
    while len(order_lst) != 1:
        for i in order_lst:
            global turn
            turn = True
            print("-----------------new player----------------------------")
            print('player', i)
            while turn:
                print("new turn player", i  )
                order_lst_mod = order_lst.copy()
                if i in order_lst_mod:
                    order_lst_mod.remove(i)
                go_fish_bool = False
                global ask_card
                ask_card = "cat"
                global turn_end
                global found_bool
                turn_end = False
                ###initialize the data structures to find which cards can give a full set and the players to ask for
                card_count_dict = defaultdict(lambda: 0)
                card_holder_dict = defaultdict(lambda:[])
                memory_lvl = memory_lst[i]
                memory_lvl_not = memory_lst_not[i]

                found_bool = False
                i_cards = card_container[i]
                if i_cards == []:
                    turn = False
                    break
                for card in i_cards:
                    card_count_dict[card] +=1
                                                                            ###gather info on what people have
                print('card count dict1', card_count_dict)

                for card in np.unique(i_cards):
                    for j in range(len(order_lst)):
                            if j!=i:
                                if card in knowledge_have[j]:
                                    found_bool = True
                                    card_count_dict[card]+=1
                                    card_holder_dict[card].append(j)
                                                                                                    ###Someone else has a card within I's hand
                print('i cards ', i_cards)
                print('knowledge have', knowledge_have)
                print('knowledge not have', knowledge_not_have)

                U = random.random()
                if found_bool:
                    print('found bool')
                    if U > memory_lvl:
                        ask_card = random.choice(i_cards)
                        print('ask_card', ask_card)
                        j = random.choice(order_lst_mod)
                        # these knowledge updates happen (ALL)
                        if ask_card in card_container[j]:
                            print('had the card ')
                            card_container[i].append(ask_card)
                            card_container[j] = [i for i in card_container[j] if i != ask_card]
                            card_count_dict[ask_card] += 1


                        else:
                            print('didnt have the card ')

                            if deck!= []:
                                print('go fish ')

                                go_fish_card = random.choice(deck)
                                go_fish_bool = True
                                deck.remove(go_fish_card)
                                card_container[i].append(go_fish_card)
                            turn = False  # comeback global
                            break  # comeback breaks
                        if not turn:
                            break
                    print('not supposed to get here')
                    #strategy is to only ask for cards when you can definitely get a 4-group
                    max_val = max(card_count_dict.values())
                    print('max_val', max_val)
                    if max_val != 4:
                        found_bool = False                          #fix break
                    print('went past')
                                                                     #indo
                    four_card_lst = []
                    for card in card_count_dict:
                        if card_count_dict[card] == 4:
                            four_card_lst.append(card)


                    for ask_card in four_card_lst:
                        for j in card_holder_dict[ask_card]:
                                                                                            #remove cards from players hand, player doesn't
                            card_container[j] = [i for i in card_container[j] if i != ask_card]

                            knowledge_have[i].add(ask_card)
                            knowledge_not_have[j].add(ask_card)
                            if ask_card in knowledge_not_have[i]:
                                knowledge_not_have[i].remove(ask_card)
                            if ask_card in knowledge_have[j]:
                                knowledge_have[j].remove(ask_card)

                        card_container[i] = [j for j in card_container[i] if j != ask_card]        #remove cards from i's hand, and remove it from the knowledge, update full set
                        full_sets[i].append(ask_card)


                else:
                    print('not found bool')#want to keep asking for cards people know that you have, to reduce their game knowledge
                    if U>memory_lvl_not:
                        ask_card = random.choice(i_cards)
                        print('ask card', ask_card)
                        j = random.choice(order_lst_mod)                   # these knowledge updates happen (ALL)
                        if ask_card in card_container[j]:
                            print('take card')
                            card_container[i].append(ask_card)
                            card_container[j] = [i for i in card_container[j] if i != ask_card]
                            card_count_dict[ask_card] += 1
                            #indoiscute
                        else:
                            if deck != []:
                                print('go fish ')
                                go_fish_card = random.choice(deck)
                                go_fish_bool = True

                                deck.remove(go_fish_card)
                                card_container[i].append(go_fish_card)
                                print('paas')
                            turn = False  # comeback global
                            break  # comeback breaks
                        if not turn:
                            break
                    print('not supposed to get here')
                    if knowledge_have[i] != set():
                        print('new start')                               #pick cards that other people know you have already
                        ask_card = random.choice(list(knowledge_have[i]))
                        print('ask card', ask_card)

                    else:
                        ask_card = random.choice(i_cards)
                        print('ask card', ask_card)

                    found_not_bool = False
                    for player in range(len(knowledge_not_have)):
                        if ask_card in knowledge_not_have[player]:
                            j = player
                            found_not_bool = True
                            break


                    if found_not_bool and deck!=[]:                                                  ###go fish
                        go_fish_card = random.choice(deck)
                        go_fish_bool = True
#####pokemon
                        turn = False
                        deck.remove(go_fish_card)
                        card_container[i].append(go_fish_card)
                    else:
                        j = random.choice(order_lst_mod)
                        if ask_card in card_container[j]:
                            card_container[i].append(ask_card)
                            card_container[j] = [i for i in card_container[j] if i != ask_card]
                            card_count_dict[ask_card] += 1

                        else:                                                          ###go fish
                            if deck != []:
                                go_fish_card = random.choice(deck)
                                go_fish_bool = True
                                deck.remove(go_fish_card)
                                card_container[i].append(go_fish_card)
                            turn = False

                    knowledge_have[i].add(ask_card)
                    knowledge_not_have[j].add(ask_card)

                    if ask_card in knowledge_not_have[i]:
                        knowledge_not_have[i].remove(ask_card)
                    if ask_card in knowledge_have[j]:
                        knowledge_have[j].remove(ask_card)
                    print('supposed to be here')
                    if card_count_dict[ask_card] == 4:


                        print('got a 4 match')
                        card_container[i] = [j for j in card_container[i] if j != ask_card]       # remove cards from i's hand, and remove it from the knowledge, update full sets


                        if ask_card in knowledge_have[i]:
                            knowledge_have[i].remove(ask_card)
                        full_sets[i].append(ask_card)
                if card_container[i] == []:
                    order_lst.remove(i)
                # print(max_val)


                print('deck size', len(deck))

        print('indo')
            # print(card_container)

        print('full sets', full_sets)

    ### get the results of the game
    result_dict = {}
    for i in range(num_players):
        player_i_full_sets = full_sets[i]
        result_dict[i] =  player_i_full_sets

    result_dict = dict(sorted(result_dict.items()))

    return list(result_dict.keys())


class go_fish():

    def __init__(self, num_players):
        """
        Initialize the queue.
        """
        self.knowledge_have = []
        self.knowledge_not_have = []
        self.full_sets = []
        self.card_container = []
        self.order_lst = []
        self.card_count_dict = defaultdict(lambda: 0)
        self.card_holder_dict = defaultdict(lambda: [])
        self.turn = True
        self.deck = []
        self.num_players = num_players

    def __len__(self):
        """
        Returns: an integer representing the number of items in the queue.
        """

        len_queue = len(self._queue)
        return len_queue

    def __str__(self):
        """
        Returns: a string representation of the queue.
        """

        return str(self._queue)

    def go_fish(self, i):

        go_fish_card = random.choice(self.deck)
        go_fish_bool = True
        self.deck.remove(go_fish_card)
        self.card_container[i].append(go_fish_card)

        self.turn = False  # comeback global

    def take_card(self, ask_card, i, j):

        self.card_container[i].append(ask_card)
        self.card_container[j] = [i for i in self.card_container[j] if i != ask_card]
        self.card_count_dict[ask_card] += 1

    def clear_guaranteed_four_sets(self, four_card_lst, i, j):
        for ask_card in four_card_lst:
            for j in self.card_holder_dict[ask_card]:
                # remove cards from players hand, player doesn't
                self.card_container[j] = [i for i in self.card_container[j] if i != ask_card]

                self.knowledge_have[i].add(ask_card)
                self.knowledge_not_have[j].add(ask_card)
                if ask_card in self.knowledge_not_have[i]:
                    self.knowledge_not_have[i].remove(ask_card)
                if ask_card in self.knowledge_have[j]:
                    self.knowledge_have[j].remove(ask_card)

            self.card_container[i] = [j for j in self.card_container[i] if
                                      j != ask_card]  # remove cards from i's hand, and remove it from the knowledge, update full set
            self.full_sets[i].append(ask_card)

    def initialize(self, num_players):
        Ranks = ['A', '2', '3', '4',
                 '5', '6', '7', '8',
                 '9', '10', 'J', 'Q', 'K']
        deck = []
        for j in range(4):
            for rank in Ranks:
                self.deck.append(rank)
        # initialize knowledge to size of players

        for i in range(num_players):
            self.knowledge_have.append(set())
            self.knowledge_not_have.append(set())
            self.full_sets.append([])

        for i in range(num_players):
            initial_cards = random.sample(deck, 5)
            for card in initial_cards:
                self.deck.remove(card)
            self.card_container.append(initial_cards)

    def get_knowledge_have(self):
        return self.knowledge_have

    def get_knowledge_not_have(self):
        return self.knowledge_not_have

    def get_full_sets(self):
        return self.full_sets

    def get_card_container(self):
        return self.card_container

    def get_order_lst(self):
        return self.order_lst

    def get_card_count_dict(self):
        return self.card_count_dict

    def get_card_holder_dict(self):
        return self.card_holder_dict

    def get_turn(self):
        return self.turn

    def get_deck(self):
        return self.deck



#
# class go_fish():
#
#     def __init__(self, num_players):
#         """
#         Initialize the queue.
#         """
#         self.knowledge_have = []
#         self.knowledge_not_have = []
#         self.full_sets = []
#         self.card_container = []
#         self.order_lst = []
#         self.turn = True
#         self.deck = []
#         self.num_players = num_players
#
#     def turn_false(self):
#         self.turn = False
#
#     def take_card_knowledge_update(self,ask_card, i,j):
#         if ask_card in self.knowledge_have[j]:
#             self.knowledge_have[j].remove(ask_card)
#         if ask_card in self.knowledge_not_have[i]:
#             self.knowledge_not_have[i].remove(ask_card)
#
#         self.knowledge_have[i].add(ask_card)
#         self.knowledge_not_have[j].add(ask_card)
#
#     def go_fish(self, i):
#
#         go_fish_card = random.choice(self.deck)
#         go_fish_bool = True
#         self.deck.remove(go_fish_card)
#         self.card_container[i].append(go_fish_card)
#         self.turn = False
#         print('go fish verify')
#         return go_fish_card
#
#     def take_card(self, ask_card, i, j,card_count_dict_actual):
#
#         len_j_before = len(self.card_container[j])
#         self.card_container[j] = [i for i in self.card_container[j] if i != ask_card]
#         #knoledge comeback
#         self.take_card_knowledge_update(ask_card,i,j)
#         len_j_after = len(self.card_container[j])
#         print('num occurents', len_j_before - len_j_after)
#         for _ in range(len_j_before - len_j_after):
#             self.card_container[i].append(ask_card)
#             print('add', i,":", self.card_container[i])
#             card_count_dict_actual[ask_card] +=1
#         print('take card verify')
#
#         return card_count_dict_actual
#
#     def produce_places(self,dog):
#         order_lst = []
#         skip_lst = []
#
#         while len(skip_lst) != len(dog):
#             max_val = -1
#             idx = None
#             for i, player in enumerate(dog):  # Using enumerate to get both index and player
#                 if i not in skip_lst:
#                     if len(player) > max_val:
#                         max_val = len(player)
#                         idx = i
#
#             if idx is not None:
#                 order_lst.append(idx)
#                 skip_lst.append(idx)
#
#         return order_lst
#
#
#
#     def check_four_groups(self,ask_card,i, card_holder_dict, card_count_dict_actual):
#         print('check four groups', card_count_dict_actual)
#         if card_count_dict_actual[ask_card] == 4:
#             print('check_four_grups')
#             self.card_container[i] = [j for j in self.card_container[i] if j != ask_card]
#             for j in card_holder_dict[ask_card]:
#                 self.card_container[j] = [i for i in self.card_container[j] if i != ask_card]
#
#             #adjust knowledge (not including the discarding)
#             for j in self.order_lst:
#                 self.take_card_knowledge_update(ask_card,i,j)
#
#             #discard the four set from knowledge
#             if ask_card in self.knowledge_have[i]:
#                 self.knowledge_have[i].remove(ask_card)
#             self.knowledge_not_have[i].add(ask_card)
#
#             self.full_sets[i].append(ask_card)
#
#     def clear_guaranteed_four_sets(self, four_card_lst, i,card_holder_dict):
#         for ask_card in four_card_lst:
#             for j in card_holder_dict[ask_card]:
#                 # remove cards from players hand, player doesn't
#                 self.card_container[j] = [i for i in self.card_container[j] if i != ask_card]
#             self.card_container[i] = [j for j in self.card_container[i] if j != ask_card]  # remove cards from i's hand, and remove it from the knowledge, update full set
#
#             #adjust knowledge (not including the discarding)
#             for j in self.order_lst:
#                 self.take_card_knowledge_update(ask_card, i, j)
#
#             # discard the four set from knowledge
#             if ask_card in self.knowledge_not_have[i]:
#                 self.knowledge_have[i].remove(ask_card)
#             self.knowledge_not_have[i].add(ask_card)
#
#             self.full_sets[i].append(ask_card)
#
#
#     def initialize(self, num_players):
#         Ranks = ['A', '2', '3', '4',
#                  '5', '6', '7', '8',
#                  '9', '10', 'J', 'Q', 'K']
#         for j in range(4):
#             for rank in Ranks:
#                 self.deck.append(rank)
#         # initialize knowledge to size of players
#
#         for i in range(num_players):
#             self.knowledge_have.append(set())
#             self.knowledge_not_have.append(set())
#             self.full_sets.append([])
#
#         for i in range(num_players):
#             initial_cards = random.sample(self.deck, 5)
#             for card in initial_cards:
#                 self.deck.remove(card)
#             self.card_container.append(initial_cards)
#
#     def get_knowledge_have(self):
#         return self.knowledge_have
#
#     def get_knowledge_not_have(self):
#         return self.knowledge_not_have
#
#     def get_full_sets(self):
#         return self.full_sets
#
#     def get_card_container(self):
#         return self.card_container
#
#     def get_order_lst(self):
#         return self.order_lst
#
#     def get_turn(self):
#         return self.turn
#
#     def get_deck(self):
#         return self.deck
#
#


def run_sim_quiet(num_players, memory_lst, memory_lst_not, order_lst):
    game_info = go_fish(num_players)
    game_info.initialize(num_players)

    # a list of all the ranks
    end_condition = []
    for i in range(num_players):
        end_condition.append([])
    card_container = "cat"
    #Play game
    while card_container != end_condition:
        for i in order_lst:
            game_info.turn = True
            # print("-----------------new player----------------------------")
            # print('player', i)
            while game_info.turn:
                knowledge_have = game_info.get_knowledge_have()
                knowledge_not_have = game_info.get_knowledge_not_have()
                full_sets = game_info.get_full_sets()
                card_container = game_info.get_card_container()
                deck = game_info.get_deck()

                # print("new turn player", i  )
                order_lst_mod = order_lst.copy()
                if i in order_lst_mod:
                    order_lst_mod.remove(i)
                go_fish_bool = False
                global ask_card
                ask_card = "cat"
                global turn_end
                global found_bool
                turn_end = False
                ###initialize the data structures to find which cards can give a full set and the players to ask for
                card_count_dict = defaultdict(lambda: 0)
                card_holder_dict = defaultdict(lambda:[])
                memory_lvl = memory_lst[i]
                memory_lvl_not = memory_lst_not[i]

                found_bool = False
                i_cards = card_container[i]
                if i_cards == []:
                    game_info.turn = False
                    break
                for card in i_cards:
                    card_count_dict[card] +=1
                                                                            ###gather info on what people have
                # print('card count dict1', card_count_dict)
                card_count_dict_actual = card_count_dict.copy()

                for card in np.unique(i_cards):
                    for j in range(len(order_lst)):
                            if j!=i:
                                if card in knowledge_have[j]:
                                    found_bool = True
                                    card_count_dict[card]+=1
                                    card_holder_dict[card].append(j)
                                                                                                    ###Someone else has a card within I's hand
                # print('i cards ', i_cards)
                # print('knowledge have', knowledge_have)
                # print('knowledge not have', knowledge_not_have)
                # print('card container', card_container)

                U = random.random()
                if found_bool:
                    # print('found bool')
                    if U > memory_lvl:
                        ask_card = random.choice(i_cards)
                        # print('ask_card', ask_card)
                        j = random.choice(order_lst_mod)
                        # print('player j:', j)
                        # these knowledge updates happen (ALL)
                        if ask_card in card_container[j]:
                            card_count_dict_actual = game_info.take_card(ask_card,i,j,card_count_dict_actual)
                            #comeback cardcount
                            game_info.check_four_groups(ask_card, i, card_holder_dict, card_count_dict_actual)

                        else:
                            # print('didnt have the card ')
                            if deck!= []:
                                go_fish_card = game_info.go_fish(i)
                                card_count_dict_actual[go_fish_card] += 1
                                game_info.check_four_groups(go_fish_card, i, card_holder_dict, card_count_dict_actual)
                            game_info.turn_false()
                            # print('deck size', len(deck))
                            # print('full sets', full_sets)

                            break  # comeback breaks
                    #strategy is to only ask for cards when you can definitely get a 4-group

                    if U < memory_lvl:
                        max_val = max(card_count_dict.values())
                        # print('max_val', max_val)
                        if max_val != 4:
                            found_bool = False
                        if found_bool:
                            if max_val == 4:
                                four_card_lst = []
                                for card in card_count_dict:
                                    # print('ask_card:', card)
                                    if card_count_dict[card] == 4:
                                        four_card_lst.append(card)

                                game_info.clear_guaranteed_four_sets(four_card_lst,i,card_holder_dict)
                            # print('full sets', full_sets)


                if found_bool == False:
                    # print('not found bool')#want to keep asking for cards people know that you have, to reduce their game knowledge
                    if U > memory_lvl_not:
                        ask_card = random.choice(i_cards)
                        # print('ask card', ask_card)
                        j = random.choice(order_lst_mod)
                        # print('player j:', j)
                        # these knowledge updates happen (ALL)
                        if ask_card in card_container[j]:
                            card_count_dict_actual=game_info.take_card(ask_card,i,j,card_count_dict_actual)
                            game_info.check_four_groups(ask_card, i,card_holder_dict,card_count_dict_actual)
                        else:
                            if deck != []:
                                go_fish_card = game_info.go_fish(i)
                                card_count_dict_actual[go_fish_card] += 1
                                game_info.check_four_groups(go_fish_card, i, card_holder_dict, card_count_dict_actual)
                            game_info.turn_false()
                            # print('deck size', len(deck))
                            # print('full sets', full_sets)
                            break  # comeback breaks


                    ####################################################
                    if U < memory_lvl_not:
                        # print('not supposed to get here')

                        if knowledge_have[i] != set():
                            # print('new start')                               #pick cards that other people know you have already
                            ask_card = random.choice(list(knowledge_have[i]))
                            # print('ask card', ask_card)

                        else:
                            ask_card = random.choice(i_cards)
                            # print('ask card', ask_card)

                        found_not_bool = False
                        for player in range(len(knowledge_not_have)):
                            if ask_card in knowledge_not_have[player]:
                                j = player
                                found_not_bool = True
                                break


                        if found_not_bool and deck!=[]:                                                  ###go fish
                            go_fish_card = game_info.go_fish(i)
                            card_count_dict_actual[go_fish_card] +=1
                            # print('go fish card',go_fish_card)
                            game_info.check_four_groups(go_fish_card,i,card_holder_dict,card_count_dict_actual)
                            game_info.turn = False
                        ##either if you can't employ the strategy because you dont know or you cannot because the pond is dry
                        else:
                            j = random.choice(order_lst_mod)
                            # print('player j:', j)
                            if ask_card in card_container[j]:
                                card_count_dict_actual = game_info.take_card(ask_card, i, j,card_count_dict_actual)
                                #problem
                                game_info.check_four_groups(ask_card, i,card_holder_dict,card_count_dict_actual)
                            else:                                                          ###go fish
                                if deck != []:
                                    # slimindo
                                    go_fish_card = game_info.go_fish(i)
                                    card_count_dict_actual[go_fish_card] += 1
                                    game_info.check_four_groups(go_fish_card, i, card_holder_dict, card_count_dict_actual)
                                game_info.turn_false()
                        # print('full sets', full_sets)
                ########################################################################
                if card_container[i] == []:
                    order_lst.remove(i)
                # print(max_val)


    print(':))))', full_sets)
    return game_info.produce_places(full_sets)
