import editdistance
import random

def edit_dis(a, b):
	return editdistance.eval(a, b)


def correction(preds, label_set):
	ret = []
	for i in range(len(preds)):
		dis_vector = [edit_dis(preds[i], j) for j in label_set]
		min_dis = min(dis_vector)
		ans_set = []
		for j in range(len(label_set)):
			if dis_vector[j] == min_dis:
				ans_set.append(label_set[j])

		ret.append(random.choice(ans_set))
	return ret