import csv
import pandas as pd
import numpy as np
import random
import sys

random.seed(0)

def Greedy(budgets, bids, queries):
    revenue = 0.0
    for q in queries:
        keys = bids[q].keys()
        highestBidder = keys[0]
        highestBid = -1
        
        c = check_budget(bids[q], budgets)
        if c != -1:
            for k in keys:
                if budgets[k] >= bids[q][k]:
                    if highestBid < bids[q][k]:
                        highestBidder = k
                        highestBid = bids[q][k]
                    elif highestBid == bids[q][k]:
                        if highestBidder > k:
                            highestBidder = k
                            highestBid = bids[q][k]
                            
            revenue += bids[q][highestBidder]
            budgets[highestBidder] -= bids[q][highestBidder]
        
    return revenue

def MSVV(rembudget, budgets, bids, queries):
    revenue = 0.0
    for q in queries:
        keys = bids[q].keys()
        highestBidder = keys[0]
        
        c = check_budget(bids[q], rembudget)
        if c != -1:
            for k in keys:
                if budgets[k] >= bids[q][k]:
                    m1 = bids[q][highestBidder]*(1 - np.exp(((budgets[highestBidder] - rembudget[highestBidder])/budgets[highestBidder])-1))
                    m2 = bids[q][k]*(1 - np.exp(((budgets[k] - rembudget[k])/budgets[k])-1))
                    if m1 < m2:
                        highestBidder = k
                    elif m1 == m2:
                        if highestBidder > k:
                            highestBidder = k
                            
            revenue += bids[q][highestBidder]
            rembudget[highestBidder] -= bids[q][highestBidder]
            
    return revenue


def Balance(budgets, bids, queries):
    revenue = 0.0
    for q in queries:
        keys = bids[q].keys()
        highestBidder = keys[0]
        
        c = check_budget(bids[q], budgets)
        if c != -1:
            for k in keys:
                if budgets[k] >= bids[q][k]:
                    if budgets[highestBidder] < budgets[k]:
                        highestBidder = k
                    elif budgets[highestBidder] == budgets[k]:
                        if highestBidder > k:
                            highestBidder = k
            
            revenue += bids[q][highestBidder]
            budgets[highestBidder] -= bids[q][highestBidder]
            
    return revenue


def check_budget(b, budgets):
	keys = b.keys()
	for k in keys:
		if budgets[k] >= b[k]:
			return 0
	return -1


def calculate_revenue(budget, bids, queries, ch):
	total_revenue = 0.0
	iters = 100
	for i in range(0,iters):
		random.shuffle(queries)
		budget1 = dict(budget)
		if ch == 1:
			revenue = Greedy(budget1, bids, queries)
		elif ch == 2:
			revenue = Balance(budget1, bids, queries)
		elif ch == 3:
			revenue = MSVV(budget1, dict(budget), bids, queries)
		else:
			revenue = 0.0
            
		total_revenue += revenue

	return total_revenue/iters


def main(ch):
	budget = dict()
	bids = dict()

	inputFile = pd.read_csv('bidder_dataset.csv')

	for i in range(0, len(inputFile)):
		a = inputFile.iloc[i]['Advertiser']
		k = inputFile.iloc[i]['Keyword']
		bv = inputFile.iloc[i]['Bid Value']
		b = inputFile.iloc[i]['Budget']
		if not (a in budget):
			budget[a] = b
		if not (k in bids):
			bids[k] = {}
		if not (a in bids[k]):
			bids[k][a] = bv

	with open('queries.txt') as f:
		queries = f.readlines()

	queries = [x.strip() for x in queries]

	r = calculate_revenue(budget, bids, queries, ch)
	print(round(r,2))
	print(round(r/sum(budget.values()),2))


if __name__ == "__main__":
	if sys.argv[1] == 'greedy':
		main(1)
	elif sys.argv[1] == 'balance':
		main(2)
	elif sys.argv[1] == 'msvv':
		main(3)