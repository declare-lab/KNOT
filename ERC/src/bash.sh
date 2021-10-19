#ERC task
#dyda_0 amd 1 as local, dyda_2 as global, dyda_3 as transfer set.

#local models
	#entropy-based
CUDA_VISIBLE_DEVICES=0 python run_local_classifier.py  -loss_fn='entropy' -dataset='../train_test_split/meld_e.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=512 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'
CUDA_VISIBLE_DEVICES=1 python run_local_classifier.py  -loss_fn='entropy' -dataset='../train_test_split/iemocap.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=512 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'
CUDA_VISIBLE_DEVICES=2 python run_local_classifier.py  -loss_fn='entropy' -dataset='../train_test_split/dyda_e_0.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=512 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'
CUDA_VISIBLE_DEVICES=3 python run_local_classifier.py  -loss_fn='entropy' -dataset='../train_test_split/dyda_e_1.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=512 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'


#pretrain global model
  #entropy-based
CUDA_VISIBLE_DEVICES=1 python run_local_classifier.py -loss_fn='entropy' -dataset='../train_test_split/dyda_e_2.pkl' -batch_size=64 -epochs=10 -lr=0.001 -max_text_len=512 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='global'


#create noisy labels from local models on transfer set
CUDA_VISIBLE_DEVICES=2 python run_noisy_labels.py -tranfer_set='../train_test_split/dyda_e_3.pkl' -batch_size=1024 -save_model_path='../noisy_labels' -lm='../saved_models/best_model_entropy_meld_e.pt' -lm='../saved_models/best_model_entropy_iemocap.pt' -lm='../saved_models/best_model_entropy_dyda_e_0.pt' -lm='../saved_models/best_model_entropy_dyda_e_1.pt' -gm='../saved_models/best_model_entropy_dyda_e_2.pt'


#check if the order in which files in -lm are written matters.
#predict distribution bias, lm and gm need to be specified as architectures could be different
CUDA_VISIBLE_DEVICES=3 python calc_dist_bias.py -batch_size=128 -lm='best_model_entropy_meld_e.pt' -lm='best_model_entropy_iemocap.pt' -lm='best_model_entropy_dyda_e_0.pt' -lm='best_model_entropy_dyda_e_1.pt' -gm='best_model_entropy_dyda_e_2.pt'


#run sinkhorn
CUDA_VISIBLE_DEVICES=3 python run_sinkhorn_confident_distill.py -loss_fn='ot' -pretrained='../saved_models/best_model_entropy_dyda_e_2.pt' -noisy_tranfer_labels='../noisy_labels/dyda_e_3.pkl' -lm='../train_test_split/meld_e.pkl' -lm='../train_test_split/iemocap.pkl' -lm='../train_test_split/dyda_e_0.pkl' -lm='../train_test_split/dyda_e_1.pkl' -lm='../train_test_split/dyda_e_2.pkl'  -batch_size=128 -epochs=20 -lr=0.001 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt'
CUDA_VISIBLE_DEVICES=2 python run_sinkhorn_confident_distill.py -loss_fn='entropy' -pretrained='../saved_models/best_model_entropy_dyda_e_2.pt' -noisy_tranfer_labels='../noisy_labels/dyda_e_3.pkl' -lm='../train_test_split/meld_e.pkl' -lm='../train_test_split/iemocap.pkl' -lm='../train_test_split/dyda_e_0.pkl' -lm='../train_test_split/dyda_e_1.pkl' -lm='../train_test_split/dyda_e_2.pkl' -batch_size=128 -epochs=20 -lr=0.001 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt'




