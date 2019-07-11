#!/usr/bin/env python

loss_total = 0.0
count = 0
with open("results.txt") as f:
	for line in f:
		line_delim = line.split(" ")
		if line_delim[0] == "iter:":
			count += 1
			print(float(line_delim[-1]))
			loss_total += float(line_delim[-1])
print(loss_total)
print(count)
print("average loss: " + str(loss_total/count))
