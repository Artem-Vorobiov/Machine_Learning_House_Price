import statistics 

my_files = ['acc_set_1.txt', 'acc_set_2.txt', 'acc_set_1_LogPrice.txt', 'acc_set_2_LogPrice.txt', \
			'acc_set_1_LogGrLivArea.txt', 'acc_set_2_LogGrLivArea.txt', 'acc_set_1_LogTotalBsmtSF.txt',\
			'acc_set_2_LogTotalBsmtSF.txt', 'acc_set_2_Log4.txt']
pool    = []
acc_set = []


def calc_accuracy(file):
	with open(file, 'r') as f:
	    for line in f:
	    	number = line.split(' = ')[1]
	    	num    = number.split('\n')[0]
	    	pool.append(float(num))
	    my_mean = statistics.mean(pool)
	return pool, my_mean

for f in my_files:
	pool, my_mean = calc_accuracy(f)
	acc_set.append(my_mean)

print('\n\tFirst accuracy:\t\t\t{} \n\tSecond accuracy:\t\t{}\
	\n\tSet_1 Log_Price accuracy:\t{} \n\tSet_2 Log_Price accuracy:\t{}\
	\n\tSet_1 LogGrLivArea accuracy:\t{} \n\tSet_2 LogGrLivArea accuracy:\t{}\
	\n\tSet_1 LogTotalBsmtSF accuracy:\t{} \n\tSet_2 LogTotalBsmtSF accuracy:\t{}\
	\n\tLog4Stats accuracy:\t\t{}'.format(acc_set[0],\
	 acc_set[1], acc_set[2],acc_set[3], acc_set[4], acc_set[5], acc_set[6], acc_set[7], acc_set[8]))

print('\n\tUsing Log functions my accuracy declining...\n')