import numpy as np
from sklearn.metrics import f1_score

# f1 score
def mean_f1(pred_start, pred_end, ans_start, ans_end):
	f1_total = np.zeros(len(pred_start))
	
	for i in range(len(pred_start)):
		if (int(pred_start[i]) > int(pred_end[i])):
			temp = pred_start
			pred_start = pred_end
			pred_end = temp
		start = min (pred_start[i], ans_start[i])
		end = max (pred_end[i], ans_end[i])
		if (i<10):
			print (start)
			print (end)
		pred_start[i] -= start
		pred_end[i] -= start
		ans_start[i] -= start
		ans_end[i] -= start

		ans = np.zeros(int(end) - int(start) + 1)
		pred = np.zeros(int(end) - int(start) + 1)

		ans[int(ans_start[i]):int(ans_end[i]) + 1] = 1
		pred[int(pred_start[i]):int(pred_end[i]) + 1]=1
		if (i<10):
			print (ans)
			print (pred)
		f1_total[i] = f1_score(ans, pred, average='micro')

	return f1_total.mean()

def my_f1_score(ans, pred):
	correct = 0
	for i in range(len(ans)):
		if ans[i] * pred[i] == 1:
			correct += 1
	if correct == 0 :
		return 0
	else:
		precision = correct/sum(pred)
		recall = correct/sum(ans)
		f1 = 2 * precision * recall / (precision + recall)
		return f1

# this will change the value of parameter
def mean_f1_2(pred_start, pred_end, ans_start, ans_end):
	f1_total = np.zeros(len(pred_start))

	for i in range(len(pred_start)):
		if (int(pred_start[i]) > int(pred_end[i])):
			temp = pred_start
			pred_start = pred_end
			pred_end = temp
		start = min (pred_start[i], ans_start[i])
		end = max (pred_end[i], ans_end[i])
		# if (i<10):
			# print (start)
			# print (end)
		pred_start[i] -= start
		pred_end[i] -= start
		ans_start[i] -= start
		ans_end[i] -= start

		ans = np.zeros(int(end) - int(start))
		pred = np.zeros(int(end) - int(start))

		ans[int(ans_start[i]):int(ans_end[i])] = 1
		pred[int(pred_start[i]):int(pred_end[i])]=1
		# if (i<10):
			# print (ans)
			# print (pred)
		f1_total[i] = my_f1_score(ans, pred)

	return f1_total.mean()

# this will change the value of parameter
def mean_f1_3(pred_start, pred_end, ans_start, ans_end):
	f1_total = np.zeros(len(pred_start))

	for i in range(len(pred_start)):
		if (int(pred_start[i]) > int(pred_end[i])):
			temp = pred_start
			pred_start = pred_end
			pred_end = temp
		start = min (pred_start[i], ans_start[i])
		end = max (pred_end[i], ans_end[i])
		# if (i<10):
			# print (start)
			# print (end)
		pred_start[i] -= start
		pred_end[i] -= start
		ans_start[i] -= start
		ans_end[i] -= start

		ans = np.zeros(int(end) - int(start))
		pred = np.zeros(int(end) - int(start))

		ans[int(ans_start[i]):int(ans_end[i])] = 1
		pred[int(pred_start[i]):int(pred_end[i])]=1
		# if (i<10):
			# print (ans)
			# print (pred)
		f1_total[i] = my_f1_score(ans, pred)

	return f1_total
