#Sentiment task

#Note: make sure all models should appear in order:
#  loc Cell phones -> loc Clothing -> loc Toys (-> glob Food)

#local models
  #entropy-based
CUDA_VISIBLE_DEVICES=1 python run_local_classifier.py  -loss_fn='entropy' -dataset='../train_test_split/loc_Cell_Phones_and_Accessories.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=200 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'
CUDA_VISIBLE_DEVICES=2 python run_local_classifier.py  -loss_fn='entropy' -dataset='../train_test_split/loc_Clothing_Shoes_and_Jewelry.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=200 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'
CUDA_VISIBLE_DEVICES=3 python run_local_classifier.py  -loss_fn='entropy' -dataset='../train_test_split/loc_Toys_and_Games.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=200 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'
  #OT-based
CUDA_VISIBLE_DEVICES=3 python run_local_classifier.py  -loss_fn='ot' -dataset='../train_test_split/loc_Cell_Phones_and_Accessories.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=200 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'
CUDA_VISIBLE_DEVICES=3 python run_local_classifier.py  -loss_fn='ot' -dataset='../train_test_split/loc_Clothing_Shoes_and_Jewelry.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=200 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'
CUDA_VISIBLE_DEVICES=3 python run_local_classifier.py  -loss_fn='ot' -dataset='../train_test_split/loc_Toys_and_Games.pkl' -batch_size=64 -epochs=10 -lr=0.0005 -max_text_len=200 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='local'


#pretrain global model
  #entropy-based
CUDA_VISIBLE_DEVICES=2 python run_local_classifier.py -loss_fn='entropy' -dataset='../train_test_split/globP_set_reviews_Food.pkl' -batch_size=2048 -epochs=5 -lr=0.0005 -max_text_len=200 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='global'
  #OT-based
CUDA_VISIBLE_DEVICES=3 python run_local_classifier.py -loss_fn='ot' -dataset='../train_test_split/globP_set_reviews_Food.pkl' -batch_size=1024 -epochs=10 -lr=0.0005 -max_text_len=200 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt' -model_type='global'


#create noisy labels from local models on transfer set
CUDA_VISIBLE_DEVICES=2 python run_noisy_labels.py -tranfer_set='../train_test_split/globF_Grocery_and_Gourmet_Food.pkl' -batch_size=1024 -save_model_path='../noisy_labels' -lm='../saved_models/best_model_entropy_loc_Cell_Phones_and_Accessories.pt' -lm='../saved_models/best_model_entropy_loc_Clothing_Shoes_and_Jewelry.pt' -lm='../saved_models/best_model_entropy_loc_Toys_and_Games.pt' -gm='../saved_models/best_model_entropy_globP_set_reviews_Food.pt'


#check if the order in which files in -lm are written matters.
#predict distribution bias, lm and gm need to be specified as architectures could be different
CUDA_VISIBLE_DEVICES=3 python calc_dist_bias.py -batch_size=128 -lm='best_model_entropy_loc_Cell_Phones_and_Accessories.pt' -lm='best_model_entropy_loc_Clothing_Shoes_and_Jewelry.pt' -lm='best_model_entropy_loc_Toys_and_Games.pt' -gm='best_model_entropy_globP_set_reviews_Food.pt'

#run sinkhorn with NO pretrained global
#pretrained global model behave as local model
CUDA_VISIBLE_DEVICES=2 python run_sinkhorn_confident_distill.py -loss_fn='ot' -noisy_tranfer_labels='../noisy_labels/globF_Grocery_and_Gourmet_Food.pkl' -lm='../train_test_split/loc_Cell_Phones_and_Accessories.pkl' -lm='../train_test_split/loc_Clothing_Shoes_and_Jewelry.pkl' -lm='../train_test_split/loc_Toys_and_Games.pkl'  -lm='../train_test_split/globP_set_reviews_Food.pkl' -batch_size=1024 -epochs=20 -lr=0.001 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt'
CUDA_VISIBLE_DEVICES=3 python run_sinkhorn_confident_distill.py -loss_fn='entropy' -noisy_tranfer_labels='../noisy_labels/globF_Grocery_and_Gourmet_Food.pkl' -lm='../train_test_split/loc_Cell_Phones_and_Accessories.pkl' -lm='../train_test_split/loc_Clothing_Shoes_and_Jewelry.pkl' -lm='../train_test_split/loc_Toys_and_Games.pkl'  -lm='../train_test_split/globP_set_reviews_Food.pkl' -batch_size=1024 -epochs=20 -lr=0.001 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt'

#run sinkhorn with pretrained global
#pretrained global model behave as local model
CUDA_VISIBLE_DEVICES=2 python run_sinkhorn_confident_distill.py -loss_fn='ot' -pretrained='../saved_models/best_model_entropy_globP_set_reviews_Food.pt' -noisy_tranfer_labels='../noisy_labels/globF_Grocery_and_Gourmet_Food.pkl' -lm='../train_test_split/loc_Cell_Phones_and_Accessories.pkl' -lm='../train_test_split/loc_Clothing_Shoes_and_Jewelry.pkl' -lm='../train_test_split/loc_Toys_and_Games.pkl'  -lm='../train_test_split/globP_set_reviews_Food.pkl' -batch_size=1024 -epochs=20 -lr=0.001 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt'
CUDA_VISIBLE_DEVICES=3 python run_sinkhorn_confident_distill.py -loss_fn='entropy' -pretrained='../saved_models/best_model_entropy_globP_set_reviews_Food.pt' -noisy_tranfer_labels='../noisy_labels/globF_Grocery_and_Gourmet_Food.pkl' -lm='../train_test_split/loc_Cell_Phones_and_Accessories.pkl' -lm='../train_test_split/loc_Clothing_Shoes_and_Jewelry.pkl' -lm='../train_test_split/loc_Toys_and_Games.pkl'  -lm='../train_test_split/globP_set_reviews_Food.pkl' -batch_size=1024 -epochs=20 -lr=0.001 -save_model_path='../saved_models' -metric='f1_macro' -result_path='../results.txt'

