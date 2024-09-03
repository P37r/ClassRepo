import gurobipy as gp
from gurobipy import GRB
import numpy as np



demand_magna = [200000, 120000, 100000, 150000, 220000, 90000]
demand_continental = [120000, 120000, 100000, 110000, 100000, 100000]
capacities = [370000, 420000, 310000, 280000, 290000, 250000]
fix_costs = 1000 * np.array([600, 650, 520, 475, 488, 425])

shipping_costs = np.array([[2.50, 1.50, 3.00, 2.75, 4.00, 4.50],
                 [1.75, 3.00, 1.50, 3.00, 2.50, 3.50],
                 [3.25, 4.00, 2.50, 3.00, 2.00, 2.50],
                 [2.00, 2.00, 2.75, 2.50, 3.75, 4.00],
                 [2.25, 3.00, 2.25, 2.50, 2.75, 3.00],
                 [3.50, 3.75, 2.50, 2.50, 2.50, 2.00]])

																										# Problem 1

model1 = gp.Model()
model1.params.LogToConsole = 0
x = {}
for i in range(6):
	for j in range(6):
		x[i, j] = model1.addVar(vtype=GRB.CONTINUOUS, name=f"x[{i},{j}]")

model1.setObjective(gp.quicksum(gp.quicksum(shipping_costs[i][j] * x[i, j] for j in range(6)) for i in range(6))
+ sum(fix_costs), GRB.MINIMIZE)


for j in range(6):
	model1.addConstr(sum(x[i, j] for i in range(3)) >= demand_magna[j])
	model1.addConstr(sum(x[i, j] for i in range(3, 6)) >= demand_continental[j])

for i in range(6):
	model1.addConstr(sum(x[i, j] for j in range(6)) <= capacities[i])

model1.optimize()
print("Problem 1 Cost:", model1.ObjVal)

mat = np.empty([6,6])
for i in range(6):
	for j in range(6):
		mat[i][j] = int(x[i, j].X)

print(mat)




																										# Problem 2

model2 = gp.Model()
model2.params.LogToConsole = 0

x = {}
for i in range(6):
	for j in range(6):
		x[i, j] = model2.addVar(vtype=GRB.CONTINUOUS, name=f"x[{i},{j}]")
y = {}
for i in range(6):
	for j in range(6):
		y[i, j] = model2.addVar(vtype=GRB.CONTINUOUS, name=f"y[{i},{j}]")



model2.setObjective(gp.quicksum(gp.quicksum(shipping_costs[i][j] * x[i, j] for j in range(6)) for i in range(6))
					+ gp.quicksum(gp.quicksum(shipping_costs[i][j] * y[i, j] for j in range(6)) for i in range(6))
					+ sum(fix_costs) + 200 * 1000, GRB.MINIMIZE)


for i in range(6):
	model2.addConstr(sum(x[i, j] + y[i, j] for j in range(6)) <= capacities[i])

for j in range(6):
	model2.addConstr(sum(x[i, j] for i in range(6)) >= demand_continental[j])
	model2.addConstr(sum(y[i, j] for i in range(6)) >= demand_magna[j])

model2.optimize()

print("Problem 2 Cost:", model2.ObjVal)
print()



def problem3_4(prob3):
	model3 = gp.Model()
	model3.params.LogToConsole = 0


	x = {}
	for i in range(6):
		for j in range(6):
			x[i, j] = model3.addVar(vtype=GRB.CONTINUOUS, name=f"x[{i},{j}]")
	y = {}
	for i in range(6):
		for j in range(6):
			y[i, j] = model3.addVar(vtype=GRB.CONTINUOUS, name=f"y[{i},{j}]")


	# Close
	w ={}
	for j in range(6):
		w[j] = model3.addVar(vtype=GRB.BINARY, name=f"y[{j}]")
	# Double
	z ={}
	for j in range(6):
		z[j] = model3.addVar(vtype=GRB.BINARY, name=f"z[{j}]")


	model3.setObjective(gp.quicksum(gp.quicksum(shipping_costs[i][j] * (x[i, j]  + y[i, j])for j in range(6)) for i in range(6))
						+ sum(fix_costs) - sum(0.8  * (w[i]-z[i]) * fix_costs[i] for i in range(6)) , GRB.MINIMIZE)


	if prob3:
		for j in range(6):
			model3.addConstr(sum(x[i, j] for i in range(6)) >= demand_continental[j])
			model3.addConstr(sum(y[i, j] for i in range(6)) >= demand_magna[j])
	else:
		for j in range(4):
			model3.addLConstr(sum(x[i, j] for i in range(6)) >= demand_continental[j])
			model3.addLConstr(sum(y[i, j] for i in range(6)) >= demand_magna[j])

		for j in range(4, 6):
			model3.addLConstr(sum(x[i, j] for i in range(6)) >= 1.3 * demand_continental[j])
			model3.addLConstr(sum(y[i, j] for i in range(6)) >= 1.3 * demand_magna[j])

	for i in range(6):
		model3.addConstr(sum(x[i, j] + y[i, j] for j in range(6)) <= capacities[i] - capacities[i] * w[i] + capacities[i] * z[i])
		model3.addConstr(w[i] + z[i] <= 1)




	model3.optimize()
	if prob3:
		print("Problem 3 Cost:", model3.ObjVal)
	else:
		print("Problem 4 Cost:", model3.ObjVal)
	print("Distribution plan for Continental")

	mat = np.empty([6,6])
	for i in range(6):
		for j in range(6):
			mat[i][j] = int(x[i, j].X)
	print(mat)

	print("Distribution plan for Magna")

	mat = np.empty([6,6])
	for i in range(6):
		for j in range(6):
			mat[i][j] = int(y[i, j].X)
	print(mat)



	print([w[i].X for i in range(6)])
	print([z[i].X for i in range(6)])



																											# Problem 3

problem3_4(True)																						# Problem 4
problem3_4(False)
