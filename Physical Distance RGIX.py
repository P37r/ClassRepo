import pandas as pd
import numpy as np
import networkx as nx
import random
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# np.set_printoptions(threshold = np.inf)
from collections import defaultdict
missing_dist = [0,2,3,7,9,12,16,18,20,24,29,30,32,35]
dist_0_data_all = np.load(r"C:\Users\peter\Downloads\rep_max_MULT_DIFF_3.npy")
border_indicator = np.load(r"C:\Users\peter\Downloads\border.npy")

# counties_file = r"C:\Users\peter\Downloads\gdf.csv"
counties_file = "C:/Users/peter/Downloads/republican_min_by_county.csv"

districts_file = "C:/Users/peter/Downloads/district_df.csv"
df_counties_orig = pd.read_csv(counties_file)
df_districts_orig = pd.read_csv(districts_file)
# df is something like tx_full_data
option = pd.set_option('display.max_rows', None)
simulations = 100
rep_control =[]
# Orestis code start

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:10:27 2024

@author: orest
"""

# Maximizing democrats subproblem
dem_or_rep = 'dem'



def counties_paths(district_A, district_B, df_counties, districts_file, lb, ub, dem_or_rep):
    # Read the CSV file into a DataFrame
    df_counties.columns = ['counties_num'] + list(df_counties.columns[1:])
    df_districts = pd.read_csv(districts_file)
    df_districts.columns = ['districts_num'] + list(df_districts.columns[1:])

    border_districts = df_districts['border_districts']
    district_num = df_districts['districts_num']

    num_of_districts = max(district_num) + 1

    if max(district_A, district_B) >= num_of_districts:
        return 'district input exceeds limit'

    # district_pairs = []
    # for i in range(num_of_districts):
    # Convert every element in border_districts into a usable format, was string.
    #   string = border_districts[i]
    #   string = string.replace(".", ",")
    #    bordering_i = np.array(ast.literal_eval(string), dtype=int)
    #    for element in range(len(bordering_i)):
    #        if element != i:
    #            pair = [i,element]
    #            district_pairs.append(pair)
    # district_pairs = np.array(district_pairs)

    # Let's do an example outside of a for loop for easier debugging

    # We do [0,1]
    if dem_or_rep == 'dem':
        # First generate a list of all neighboring counties of counties in district 0, and another in district 1
        district_0_counties = df_counties.loc[df_counties['dem_district'] == district_A, 'border_counties'].tolist()

        # get the unique counties that are in district 1 and are neighboring counties to district 1 but outside district one
        list_of_lists = [eval(lst_str) for lst_str in district_0_counties]
        flat_list = [item for sublist in list_of_lists for item in sublist]
        neighbors_district_0 = np.unique(flat_list)

        # get the list of all counties in district 1
        counties_district_1 = df_counties.loc[df_counties['dem_district'] == district_B, 'counties_num'].tolist()
        counties_district_1 = np.array(counties_district_1)

        # The below are the set of target counties in the graph that we will built up later.
        # They are the nodes/ counties of district 1 that border the state line of district 0
        district_1_border_counties = np.intersect1d(counties_district_1, neighbors_district_0)
        if len(district_1_border_counties) < 1:
            return 'empty'
        else:

            # The below are the starting points from district 0.
            district_0_counties = df_counties.loc[df_counties['dem_district'] == district_A, 'counties_num'].tolist()
            district_0_counties = np.array(district_0_counties)

            district_0_neighbors = df_counties.loc[df_counties['dem_district'] == district_A, 'border_counties']
            district_0_num = df_counties.loc[df_counties['dem_district'] == district_A, 'counties_num']
            # print(district_0_neighbors)
            # print(district_0_num)

            G = nx.Graph()

            # add all the nodes
            G.add_nodes_from(district_0_num)
            G.add_nodes_from(district_1_border_counties)

            # print(len(district_1_border_counties))

            for node in G.nodes():
                # nghbs will be the list of neighbors of node
                nghbs = df_counties.loc[df_counties['counties_num'] == node, 'border_counties'].tolist()
                array_elements = nghbs[0][1:-1].split(', ')
                array_elements = [int(x) for x in array_elements]
                nghbs = np.array(array_elements)
                for node1 in G.nodes():
                    if node1 in nghbs:
                        G.add_edge(node, node1, length=1)

            # Our graph is all set up. Now we have to get a list of minimum paths with
            # constant source i.e the set of nodes in district 1, for each target node in district 0.
            minimum_paths_to_district_1 = nx.multi_source_dijkstra_path_length(G, district_1_border_counties.tolist(),
                                                                               cutoff=None, weight='weight')

            # Decide which ones to filter out
            minimum_paths_to_district_1 = {key: value for key, value in minimum_paths_to_district_1.items() if
                                           value >= lb and value <= ub}
            return minimum_paths_to_district_1
    else:
        # First generate a list of all neighboring counties of counties in district 0, and another in district 1
        district_0_counties = df_counties.loc[df_counties['rep_district'] == district_A, 'border_counties'].tolist()

        # get the unique counties that are in district 1 and are neighboring counties to district 1 but outside district one
        list_of_lists = [eval(lst_str) for lst_str in district_0_counties]
        flat_list = [item for sublist in list_of_lists for item in sublist]
        neighbors_district_0 = np.unique(flat_list)

        # get the list of all counties in district 1
        counties_district_1 = df_counties.loc[df_counties['rep_district'] == district_B, 'counties_num'].tolist()
        counties_district_1 = np.array(counties_district_1)

        # The below are the set of target counties in the graph that we will built up later.
        # They are the nodes/ counties of district 1 that border the state line of district 0
        district_1_border_counties = np.intersect1d(counties_district_1, neighbors_district_0)
        if len(district_1_border_counties) < 1:
            return 'empty'
        else:

            # The below are the starting points from district 0.
            district_0_counties = df_counties.loc[df_counties['rep_district'] == district_A, 'counties_num'].tolist()
            district_0_counties = np.array(district_0_counties)

            district_0_neighbors = df_counties.loc[df_counties['rep_district'] == district_A, 'border_counties']
            district_0_num = df_counties.loc[df_counties['rep_district'] == district_A, 'counties_num']

            G = nx.Graph()

            # add all the nodes
            G.add_nodes_from(district_0_num)
            G.add_nodes_from(district_1_border_counties)

            print('dddddddd')
            print(len(district_1_border_counties))

            for node in G.nodes():
                # nghbs will be the list of neighbors of node
                nghbs = df_counties.loc[df_counties['counties_num'] == node, 'border_counties'].tolist()
                array_elements = nghbs[0][1:-1].split(', ')
                array_elements = [int(x) for x in array_elements]
                nghbs = np.array(array_elements)
                for node1 in G.nodes():
                    if node1 in nghbs:
                        G.add_edge(node, node1, length=1)

            # Our graph is all set up. Now we have to get a list of minimum paths with
            # constant source i.e the set of nodes in district 1, for each target node in district 0.
            minimum_paths_to_district_1 = nx.multi_source_dijkstra_path_length(G, district_1_border_counties.tolist(),
                                                                               cutoff=None, weight='weight')

            # Decide which ones to filter out
            minimum_paths_to_district_1 = {key: value for key, value in minimum_paths_to_district_1.items() if
                                           value >= lb and value <= ub}
            return minimum_paths_to_district_1


def run_func(df_counties, districts_file):

    df_counties.columns = ['counties_num'] + list(df_counties.columns[1:])
    df_districts = pd.read_csv(districts_file)
    df_districts.columns = ['districts_num'] + list(df_districts.columns[1:])

    border_districts = df_districts['border_districts']
    district_num = df_districts['districts_num']

    num_of_districts = max(district_num) + 1
    # We will store all data from all pairs of districts in a dictionary. We will have to
    # convert to strings
    path_dict = {}

    for i in range(num_of_districts):
        for j in range(num_of_districts):
            pair = np.array([i, j])
            pair_key = np.array2string(pair)
            path = counties_paths(i, j, df_counties, districts_file, 1, 1, dem_or_rep)
            path_dict[pair_key] = path

    path_dict = {key: value for key, value in path_dict.items() if isinstance(value, dict) and value}

    # Path dict shows us which counties of district i have the shortest distance to district 2.

    # from path dict get the same dictionary but with the length of the key

    num_path_dict = {key: len(value) for key, value in path_dict.items()}

    prob_matrix = np.zeros((num_of_districts, num_of_districts))

    for i in range(num_of_districts):
        for j in range(num_of_districts):
            if i != j:
                prefix = f'[{i} {j}]'
                if prefix in num_path_dict:
                    matching_keys = [key for key in num_path_dict.keys() if key.startswith(prefix)]
                    matching_values = [num_path_dict[key] for key in matching_keys]
                    val = matching_values[0]
                    prob_matrix[i, j] = int(val)
    for i in range(num_of_districts):
        prob_matrix[i, i] = sum(prob_matrix[i, j] for j in range(num_of_districts))

    # Normalize
    row_sums = prob_matrix.sum(axis=1)

    # Divide each element by its corresponding row sum
    prob_matrix = prob_matrix / row_sums[:, np.newaxis]

    return prob_matrix, path_dict
# Orestis code end




def dist(neigh_districts):
    graph_data = pd.read_csv(r"C:\Users\peter\Downloads\district_df.csv")
    coord_lst = []
    for i in graph_data['centroid']:
        coord_lst.append(np.array(eval(i)))
    num_counties = len(coord_lst)
    dist_mat = np.zeros((num_counties, num_counties))
    for i in range(len(coord_lst)):
        val_i = coord_lst[i]
        for j in range(len(coord_lst)):

            val_j = coord_lst[j]
            if j in neigh_districts[i]:

                dist_mat[i][j] = np.linalg.norm(val_i - val_j)
            else:
                dist_mat[i][j] = 0
    prob_dist_mat = np.zeros((num_counties, num_counties))
    for i in range(len(coord_lst)):
        row = dist_mat[i]
        row_sum = np.sum(row)

        for j in range(len(coord_lst)):
            if j in neigh_districts[i]:
                prob_dist_mat[i][j] = row_sum / dist_mat[i][j]

        row_sum2 = np.sum(prob_dist_mat[i])
        prob_dist_mat[i][i] = row_sum2

        for j in range(len(coord_lst)):
            prob_dist_mat[i][j] = prob_dist_mat[i][j] / (row_sum2)
    return prob_dist_mat


def func(df, graph_prob,case):
    #First generate the number of districts in our current state partition
    #Make a list ‘dist_list’ of the district column (1 entry for each county)

    numDist = int(max(df['district'])) + 1
    num_counties = df.shape[0]

    # 2.Generate the total democrat/republican split population for each district.
    Dist_rep_pop = []
    Dist_dem_pop = []
    for i in range(numDist):
        df_dist = df[df['district'] == i]
        repub_dist_count = sum(df_dist['republican'])
        democrat_dist_count = sum(df_dist['democrat'])

        Dist_rep_pop.append(repub_dist_count)
        Dist_dem_pop.append(democrat_dist_count)
    # switch = Dist_rep_pop
    # if case == 5:
    #     counties_switch_out = [2,3,4,5]
    #     for i in counties_switch_out:
    #         switch[1] += 0.8 * switch[i]
    #         switch[i] -= 0.8 * switch[i]
    # if case ==6:
    #     counties_switch_out = [7, 8, 2, 5]
    #     for i in counties_switch_out:
    #         switch[1] += 0.8 * switch[i]
    #         switch[i] -= 0.8 * switch[i]
    #
    # if case == 7:
    #     counties_switch_out = [1, 3, 5, 8]
    #     for i in range(3):
    #         switch[i] += 0.8 * switch[counties_switch_out[i]]
    #     for i in counties_switch_out:
    #         switch[i] -= 0.8 * switch[i]
    # if case ==8:
    #     counties_switch_out = [2, 4, 6, 8]
    #     for i in range(3):
    #         switch[i] += 0.8 * switch[counties_switch_out[i]]
    #     for i in counties_switch_out:
    #         switch[i] -= 0.8 * switch[i]
    #
    # if case == 9:
    #
    #     counties_switch_out = [1, 3, 8, 2]
    #     for i in range(4):
    #         switch[i] += 0.8 * switch[counties_switch_out[i]]
    #     for i in counties_switch_out:
    #         switch[i] -= 0.8 * switch[i]




    a = np.array(Dist_rep_pop)
    b = np.array(Dist_dem_pop)
    a_b = a+b
    print('total pop', a_b)
    print('rep pop', a)
    print('rep perc', a/a_b)

    #error part
    # for i in missing_dist:
    #     Dist_rep_pop[i] = 2000000
    #     Dist_dem_pop[i] = 2000000
    #3.Generate the democrat/republican split population for each district. It will be used later on to compare
    # with the final result after the simulation has been completed to derive the index

    Dist_rep_percent = np.empty([numDist,1])
    Dist_dem_percent = np.empty([numDist,1])

    for i in range(numDist):
        Dist_rep_percent[i] = Dist_rep_pop[i] / (Dist_rep_pop[i] + Dist_dem_pop[i])
        Dist_dem_percent[i] = 1 - Dist_rep_percent[i]

    #4.Figure out which districts are bordering the state line. We should have data that indicates whether a
    # county borders the state (here I will be assuming that if the county borders the state it has value 1 and
    # otherwise it has value 0). Bordering_list will be an array/list of the districts that border the state.

    df_subset_border_counties = df[df['border_district'] == 1]



    #adds the county number
    border_counties = np.unique(list(df_subset_border_counties.index.values))
    border_districts = np.unique(list(df_subset_border_counties['district']))
    border_districts = list(border_districts)

    #error part
    # border_districts.append(8)
    # border_districts.append(15)
    #5. Generate the districts that neighbor each other.

    neighbor_dist = []
    for i in range(numDist):
        dist_neighbor_counties = set()
        df_dist = df[df['district'] == i]
        for neighbors in df_dist['border_districts']:
            dist_neighbor_counties.update(neighbors)

        #makes a subset of the dataframe with only the county rows from (dist_neighbor_counties) and then extract districts
        df_neighbor = df.iloc[list(dist_neighbor_counties),:]
        neigh_districts = list(np.unique(df_neighbor['district']))
        neighbor_dist.append(neigh_districts)
    #generate probability matrix
    prob = np.zeros((numDist, numDist))
    for i in range(numDist):
        for j in range(numDist):
            j_neighbor_of_i = j in neighbor_dist[i]
            is_a_border_district = i in border_districts
            # if i ==j :
            #     print('district num', i)
            #     print('neighbor condition:', j_neighbor_of_i)
            #     print('border ',is_a_border_district)

            if i ==j and is_a_border_district:
                prob[i,j] = 1/2
            elif i == j and not is_a_border_district:
                prob[i, j] = 1/(len(neighbor_dist[i]))
            elif(j_neighbor_of_i) and is_a_border_district:
                prob[i,j] = (1/2) / (len(neighbor_dist[i]) -1)
            elif j_neighbor_of_i and not is_a_border_district:
                prob[i,j] = 1/(len(neighbor_dist[i]))


    prob_dist_mat= dist(neighbor_dist)
    # physical distance adjustment
    # prob_old = prob.copy()
    # for i in range(numDist):
    #     for j in range(numDist):
    #         prob[i][j] = prob[i][j] * prob_dist_mat[i][j]
    #         graph_prob[i][j] = graph_prob[i][j] * prob_dist_mat[i][j]
    #
    # for i in range(numDist):
    #     row_sum = sum(prob[i])
    #     row_sum2 = sum(graph_prob[i])
    #     for j in range(numDist):
    #         prob[i][j] = prob[i][j] / row_sum
    #         graph_prob[i][j] = graph_prob[i][j]/ row_sum2
    return prob, Dist_dem_percent, Dist_rep_percent, numDist, Dist_rep_pop, Dist_dem_pop, graph_prob

def simulation_helper(time_iter, numDist,Dist_rep_pop, Dist_dem_pop,prob,group,Dist_dem_percent, Dist_rep_percent):
    # Keep copies after each loop you wanna start fresh
    Dist_rep_pop_loop = Dist_rep_pop
    Dist_dem_pop_loop = Dist_dem_pop
    ### row 1 is dem, row2 is rep
    storage = np.zeros((2, numDist))
    for yo in range(time_iter):
        total_pop_loop = Dist_dem_pop_loop + Dist_rep_pop_loop
        group_loop = np.floor(total_pop_loop/time_iter)
        for i in range(numDist):
            if Dist_rep_pop_loop[i] > group_loop[i] and Dist_dem_pop_loop[i] > group_loop[i]:
                U_party = random.random()
                U_transition = random.random()
                current_percentage = Dist_rep_pop_loop[i] / (Dist_rep_pop_loop[i] + Dist_dem_pop_loop[i])
                #print(current_percentage)
                #print(yo)
                #print(i)
                # if current_percentage <.2:
                #     print([i, current_percentage, Dist_rep_pop[i]], Dist_dem_pop[i])
                if current_percentage < 0:
                    print("ERROR!")
                prob_row = prob[i, :]
                # determine the next state to transition to
                for idx in range(1, len(prob_row) + 1):
                    if U_transition < sum(prob_row[:idx]):
                        idx = idx-1
                        break
                if U_party < current_percentage:
                    Dist_rep_pop_loop[i]-= group_loop[i]
                    storage[1, idx] += group_loop[i]
                else:
                    Dist_dem_pop_loop[i] -= group_loop[i]
                    storage[0, idx] += group_loop[i]
                # print(Dist_dem_pop_loop)
                #print(Dist_dem_pop)
    for i in range(numDist):
        Dist_rep_pop_loop[i] = Dist_rep_pop_loop[i]+storage[1,i]
        Dist_dem_pop_loop[i] = Dist_dem_pop_loop[i]+storage[0,i]
    print('this is storage')
    print(storage)
    init_dem_perc = np.array(Dist_dem_percent)
    init_rep_perc = np.array(Dist_rep_percent)

    current_dem_perc = np.array(Dist_dem_pop_loop) / (np.array(Dist_dem_pop_loop) + np.array(Dist_rep_pop_loop))
    current_rep_perc = 1 - current_dem_perc


    rgix_rep = []
    for i in range(len(init_dem_perc)):
        rgix_rep.append(float((init_rep_perc[i] - current_rep_perc[i]) * 100))

    rgix_dem = []
    for i in range(len(init_dem_perc)):
        rgix_dem.append(float((init_dem_perc[i] - current_dem_perc[i]) * 100))
    # print('rep_rgix', rgix_rep)
    # print('dem_rgix', rgix_dem)
    return np.array(rgix_rep), np.array(rgix_dem)
#
# def run_sim(trials,time_iter, numDist,Dist_rep_pop, Dist_dem_pop,prob,group):
#     dem_result_array = np.zeros(numDist)
#     rep_result_array = np.zeros(numDist)
#
#     var_dict_rep = defaultdict(lambda: [])
#     var_dict_dem = defaultdict(lambda: [])
#
#     for trial in range(trials):
#         # print(trial)
#         Dist_rep_pop2 = Dist_rep_pop.copy()
#         Dist_dem_pop2 = Dist_dem_pop.copy()
#         rgix_rep , rgix_dem = simulation_helper(time_iter, numDist,Dist_rep_pop2, Dist_dem_pop2,prob,group,Dist_dem_percent, Dist_rep_percent)
#         empty[trial] = rgix_rep
#         for i in range(len(rgix_rep)):
#             var_dict_rep[i].append(rgix_rep[i])
#         for i in range(len(rgix_dem)):
#             var_dict_dem[i].append(rgix_dem[i])
#         dem_result_array += rgix_dem
#         rep_result_array += rgix_rep
#     for i in var_dict_rep:
#         var_dict_rep[i] = np.var(list(var_dict_rep[i]))
#         var_dict_dem[i] = np.var(list(var_dict_dem[i]))
#     return rep_result_array/trials, dem_result_array/ trials, list(var_dict_rep.values()), list(var_dict_dem.values()),empty


df = pd.read_csv(r"C:\Users\peter\Downloads\TX_DATA.csv")

df_arizona = pd.read_csv(r"C:\Users\peter\Downloads\az_dist_results.csv")

df_new = pd.read_csv(r"C:\Users\peter\Downloads\district_df.csv")


df_arizona.rename(columns = {'border':'border_district'}, inplace = True)
df_arizona.rename(columns = {'neighbors':'border_districts'}, inplace = True)
df_arizona['district'] = df_arizona['district'] - 1

df_new['border_district']  = df_arizona['border_district']


# DIVIDER



#
# # CHANGES IN SIMULATION HELPER
#
# #1. Group size is a vector
#
# #2. Work with copies Dist_rep_pop_loop of Dist_rep_pop as otherwise  the 'looper' function will read the already changed dist_rep_pop per iteration
#
# #3. After each iteration calculate again the group sizes to account for changes in population (I don't think it makes a huge difference)
#
# #4. Perform the algorithm only if current district size> group_size. This will help with errors


def simulation_helper(time_iter, numDist,Dist_rep_pop, Dist_dem_pop,prob,group,Dist_dem_percent, Dist_rep_percent):
    # Keep copies after each loop you wanna start fresh
    Dist_rep_pop_loop = Dist_rep_pop
    Dist_dem_pop_loop = Dist_dem_pop

    for yo in range(time_iter):
        total_pop_loop = Dist_dem_pop_loop + Dist_rep_pop_loop
        group_loop = np.floor(total_pop_loop/time_iter)
        for i in range(numDist):
            if Dist_rep_pop_loop[i] > group_loop[i] and Dist_dem_pop_loop[i] > group_loop[i]:
                U_party = random.random()
                U_transition = random.random()
                current_percentage = Dist_rep_pop_loop[i] / (Dist_rep_pop_loop[i] + Dist_dem_pop_loop[i])
                #print(current_percentage)
                #print(yo)
                #print(i)
                # if current_percentage <.2:
                #     print([i, current_percentage, Dist_rep_pop[i]], Dist_dem_pop[i])
                if current_percentage < 0:
                    print("ERROR!")
                prob_row = prob[i, :]
                # determine the next state to transition to
                for idx in range(1, len(prob_row) + 1):
                    if U_transition < sum(prob_row[:idx]):
                        idx = idx-1
                        break
                if U_party < current_percentage:
                    Dist_rep_pop_loop[i]-= group_loop[i]
                    Dist_rep_pop_loop[idx] += group_loop[i]
                else:
                    Dist_dem_pop_loop[i] -= group_loop[i]
                    Dist_dem_pop_loop[idx] += group_loop[i]
                # print(Dist_dem_pop_loop)
                #print(Dist_dem_pop)
    init_dem_perc = np.array(Dist_dem_percent)
    init_rep_perc = np.array(Dist_rep_percent)

    current_dem_perc = np.array(Dist_dem_pop_loop) / (np.array(Dist_dem_pop_loop) + np.array(Dist_rep_pop_loop))
    current_rep_perc = 1 - current_dem_perc


    rgix_rep = []
    for i in range(len(init_dem_perc)):
        rgix_rep.append(float((init_rep_perc[i] - current_rep_perc[i]) * 100))

    rgix_dem = []
    for i in range(len(init_dem_perc)):
        rgix_dem.append(float((init_dem_perc[i] - current_dem_perc[i]) * 100))
    # print('rep_rgix', rgix_rep)
    # print('dem_rgix', rgix_dem)
    return np.array(rgix_rep), np.array(rgix_dem)
def looper(simulations, time_iter, numDist,Dist_rep_pop, Dist_dem_pop,prob,group,Dist_dem_percent, Dist_rep_percent):
    final_array = np.zeros(numDist)
    for i in range(simulations):
        [rgix_rep,rgix_dem] = simulation_helper(time_iter, numDist,Dist_rep_pop, Dist_dem_pop,prob,group,Dist_dem_percent, Dist_rep_percent)
        final_array = final_array + rgix_rep
        empty[i] = rgix_rep
    rgix_rep = final_array/simulations
    rgix_dem = -rgix_rep
    return rgix_rep, rgix_dem


probability_physical_distance = 0
# _________________________________________________
for case in range(10):
    print()
    print()
    print()
    print()
    print()


    print("THIS OUTPUT IS FOR CASE" + str(case))
    print("_____________________________________________________________________________________________________")
    print("_____________________________________________________________________________________________________")
    print("_____________________________________________________________________________________________________")
    print("_____________________________________________________________________________________________________")
    dist_0_data = np.load(r"C:\Users\peter\OneDrive\Desktop\RGIX Cases\district_assignments_"+str(case)+".npy")
    # dist_0_data = dist_0_data_all[:, case]
    # print(dist_0_data)



    df_counties = df_counties_orig.copy()
    df_districts = df_districts_orig.copy()
    df_counties.columns = ['counties_num'] + list(df_counties.columns[1:])

    df_counties['dem_district'] = dist_0_data

    df_districts.columns = ['districts_num'] + list(df_districts.columns[1:])

    border_districts = df_districts['border_districts']
    district_num = df_districts['districts_num']

    num_of_districts = max(district_num) + 1
    # Added this
    df_counties['border_indicator'] = border_indicator
    district_border_indicator_lst = []
    democrat_pop = []
    republican_pop = []
    for i in range(num_of_districts):
        sub_df = df_counties[df_counties['dem_district'] == i]

        dem_pop_dist_i = sum(sub_df['democrat'])
        rep_pop_dist_i = sum(sub_df['republican'])
        democrat_pop.append(dem_pop_dist_i)
        republican_pop.append(rep_pop_dist_i)
        if 1 in list(sub_df['border_indicator']):
            district_border_indicator_lst.append(1)
        else:
            district_border_indicator_lst.append(0)
    #______________________________________________________
    #------------------------------ CHANGED ------------------------------------------#

    # The data used to call the functions below

    # PROBABILITY MATRIX( Peter's physical distance + Orestis graph) !!! ONLY THING THAT SHOULD BE DIFFERENT

    #RACCOON
    # Peter edit ____________________________________________________________
    graph_prob , dist_dict= run_func(df_counties,districts_file)
    # print(df_new['border_districts'][0])
    # print(type(df_new['border_districts'][0]))

    # print(df_new)
    # print(df_new['border_districts'][0])
    # print(type(df_new['border_districts'][0]))
    dist_neigh_dict = defaultdict(lambda: [])
    for pair in dist_dict:
        dist_neigh_dict[int(pair[1])].append(int(pair[3]))

    # print('dist neigh', dist_neigh_dict)

    for i in range(len(df_new['district'])):
        df_new['border_districts'][i] = dist_neigh_dict[i]


    # Peter edit
    df_new['border_district'] = district_border_indicator_lst

    # print("THIS IS IMPORTANT", df_new[['border_districts']])

    df_new['republican'] = republican_pop
    df_new['democrat'] = democrat_pop
    prob, dist_dem_percent, dist_rep_percent, numDist, dist_rep_pop, dist_dem_pop,mod_graph_prob = func(df_new,graph_prob,case)
    # prob_mat_grapph = run_func(counties_file, districts_file)
    # print('prob graph shape',prob_mat_grapph)

    #error part
    for i in range(len(prob)):
        if len(np.unique(prob[i])) == 1:
            new_row = np.zeros([len(prob),1])
            new_row[len(prob)-1] = 1
            prob[i] = np.squeeze(new_row)


    row_sums = np.sum(prob, axis=1)


    time_iter = 3000
    empty = np.empty((simulations, num_of_districts))



    probability_physical_distance = prob.copy()


    # group !!!! CHANGED TO AN NP ARRAY AS WE WANT DIFFERENT GROUP SIZES FOR EACH DISTRICT
    dist_dem_pop = np.array(dist_dem_pop)
    dist_rep_pop = np.array(dist_rep_pop)
    dist_rep_percent = np.array(dist_rep_percent)
    dist_dem_percent = np.array(dist_dem_percent)
    total_pop = dist_dem_pop + dist_rep_pop
    group = np.floor(3 * total_pop / time_iter)
    # print(group)



    print('probability matrix', prob)


    [rgix_rep, rgix_dem] = looper(simulations, time_iter, num_of_districts, dist_rep_pop, dist_dem_pop,
                                  prob, group, dist_dem_percent, dist_rep_percent)


    sample_mean = np.mean(empty, axis=0)
    sample_variance = np.var(empty, axis=0)
    percent_after = dist_rep_percent.reshape(-1) - sample_mean/100
    rep_num = 0
    print('percent after', percent_after)
    for i in percent_after:
        if i>0.5:
            rep_num+=1
    rep_control.append(rep_num)
    print('sample mean:', sample_mean)
    print('sample variance:', sample_variance)
    print('mean_square_error')
    value = np.sum(np.square(sample_mean)) / len(sample_mean)
    print(value)
    print('minimum')
    print(min(sample_mean))
    print('maximum')
    print(max(sample_mean))

    # Calculate standard error of the mean (SEM)
    sem_vector = np.std(empty, axis=0) / np.sqrt(empty.shape[0])

    # Define confidence levels
    confidence_levels = [0.90, 0.95]

    # Calculate Z-scores for the given confidence levels
    z_scores = [norm.ppf((1 + level) / 2) for level in confidence_levels]

    # Calculate the margin of error for each confidence level
    margins_of_error = [z * sem_vector for z in z_scores]

    # Calculate confidence intervals
    confidence_intervals = [(sample_mean - margin, sample_mean + margin) for margin in margins_of_error]

    # Display results
    for i, (lower, upper) in enumerate(confidence_intervals):
        print(f"{int(confidence_levels[i] * 100)}% Confidence Intervals:")
        for j, (l, u) in enumerate(zip(lower, upper)):
            print(f"Entry {j + 1}: [{l}, {u}]")
        print()

print('rep control', rep_control)



