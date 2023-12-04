''' Train '''
# 1: Pattern A - Fixed 
python main_train.py --S_type A --gpu_ids 0

# 2: Pattern D - Fixed 
python main_train.py --S_type D --gpu_ids 1

# 3: Pattern A - Optimized
python main_train.py --S_type A --S_learn --gpu_ids 2

# 4: Pattern D - Optimized
python main_train.py --S_type D --S_learn --gpu_ids 3

''' Test '''
python main_test.py --exp 1 --save --sigmas 0 --data_test Kodak24
python main_test.py --exp 2 --save --sigmas 10 --data_test McMaster