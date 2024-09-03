import gurobipy as gp
from gurobipy import GRB
import numpy as np


# Making the model to find the optimal distribution ____________________________________________________________________

plant_to_brewery2 = np.array([
    [0.026, 0.017, 0.02, 0.019, 0.032],
    [0.037, 0.017, 0.031, 0.03, 0.022],
    [0.032, 0.033, 0.004, 0.028, 0.048]
])





plant_to_brewery = np.array([
    [0.026, 0.017],
    [0.037, 0.017],
    [0.032, 0.033]
])

brewery_to_DC = np.array([
    [0, 0.04, 0.052, 0.017, 0.055, 0.042],
    [0.032, 0.041, 0.039, 0.027, 0.023, 0.043]
])

# cat 0.018000000000000002
# dog 0.038

# 1,3
# 0,2

capacities_malt = [30,68,20]
capacities_breweries = [220,200]
Demand = [103, 74, 50, 60, 102, 13]

malt_yield = [8.333, 8.333, 9.091]

model = gp.Model("P_median")

x = {}
for i in range(2):
    for j in range(6):
        x[i,j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x[{i}, {j}]")

# Define y_ij variables
y = {}
for i in range(3):
    for j in range(2):
        y[i, j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y[{i},{j}]")

# Constraints
                                                            # Capacity Constraints
for i in range(3):
    model.addConstr(capacities_malt[i] >= gp.quicksum(y[i,j] for j in range(2)))

for j in range(2):
    model.addConstr(capacities_breweries[j] >= gp.quicksum(x[j,k] for k in range(6)))


                                                            #Constraints to Fulfill demand
for k in range(6):
    model.addConstr(Demand[k] <= gp.quicksum(x[j, k] for j in range(2)))
                                                            #Flow Constaint
for j in range(2):
    model.addConstr(gp.quicksum(malt_yield[i] * y[i, j] for i in range(3)) == gp.quicksum(x[j, k] for k in range(6)))

# Objective Function
model.setObjective(gp.quicksum(gp.quicksum( brewery_to_DC[i][j] * x[i,j] for j in range(6)) for i in range(2)) +
                    gp.quicksum(gp.quicksum( plant_to_brewery[i][j] * y[i,j] for j in range(2)) for i in range(3))
                   , GRB.MINIMIZE)

# Optimize the model
model.optimize()
print('lol', x[1,3].SAObjLow)
print('lol2',x[0,2].SAObjLow)

print('cat', y[2,0].SAObjLow)
print('dog',y[2,1].SAObjLow)

# Print the optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    for v in model.getVars():
        print(f"{v.varName}: {v.x}")
    print(f"Objective value: {model.objVal}")
else:
    print("Optimization did not converge")


# Getting the objective value for the current plan ____________________________________________________________________


y_curr = np.array([
    [0, 24],
    [2.42, 0],
    [20,0]
])

x_curr = np.array([
    [103,49,50,0,0,0],
    [0,25,0,60,102,13]
])
curr_obj_val = gp.quicksum(gp.quicksum( brewery_to_DC[i][j] * x_curr[i,j] for j in range(6)) for i in range(2)) + \
gp.quicksum(gp.quicksum( plant_to_brewery[i][j] * y_curr[i,j] for j in range(2)) for i in range(3))

print('curr obj val', curr_obj_val)

