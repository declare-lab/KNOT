'''
Emotion label to id-
anger:		0 
disgust:	1 
fear:		2 
happiness:	3 
no emotion:	4 
sadness:	5 
surprise:	6

-huggingface datasets have the same format for DailyDialog and MELD
-need to add labels from  IEMOCAP

Excitement: 7
Frustration: 8
Other: 9
Unknown: 10
'''

#The coordinates for above emotions based on circumplex space model are:
#[-0.4,0.8], [-0.7,0.5], [-0.1,0.8], [0.9,0.2], [0.0,0.0], [-0.9,-0.4], [0.4,0.9],[0.7, 0.7], [-0.6,0.4], [100.0,100.0], [-100.0,-100.0] 
#link: paper Asymmetrical Facial Expressions based on an Advanced Interpretation of Two-dimensional Russells Emotional Model

#pip install datasets

from datasets import load_dataset

import pickle as pk

for data_name in ['dyda_e', 'iemocap', 'meld_e']:
	print(f"\n\nWorking on {data_name} data...")
	dataset = load_dataset('silicone', data_name)

	texts = dict()
	labels = dict()

	for dat_type in ['train', 'test', 'validation']:
		print('Number of conversations: ', len(set(dataset[dat_type]['Dialogue_ID'])))
		#
		all_utterances = dataset[dat_type]['Utterance']
		all_ids = dataset[dat_type]['Dialogue_ID']
		#
		id_group = None
		conversation = []
		all_conversations = []
		for item, utt in enumerate(all_utterances):
			if all_ids[item] != id_group:
				all_conversations.append(conversation)
				id_group = all_ids[item]
				conversation = [utt]
			else:
				conversation.append(utt)
		#
		all_conversations.append(conversation)
		all_conversations = all_conversations[1:]
		#
		#I used the following snippet to chose number of past conversations for a sample (on DD)
		#max([sum([len(i.split()) for i in conv[:8]]) for conv in all_conversations])
		max_context = 8
		#
		utt_context_list = []
		for item in all_conversations:
			running_context = []
			for utt in item:
				running_context.append(utt)
				running_context = running_context[-max_context:]
				utt_context = [running_context[-i] for i in range(1,len(running_context)+1)]
				utt_context_list.append(' [SEP] '.join(utt_context))
		#
		texts[dat_type] = utt_context_list
		if data_name == 'iemocap':
			print("Special label processing for IEMOCAP...")
			#lab_map = {0:0, 1:1, 2:7, 3:2, 4:8, 5:3, 6:4, 7:9, 8:5, 9:6, 10:10}
			lab_mapA = {0:0, 1:1, 2:7, 3:2, 4:8, 5:3, 6:4, 7:9, 8:5, 9:6, 10:10}
			lab1_to_lab2_mapA = lambda lab_list: [lab_mapA[lab] for lab in lab_list]
			labels[dat_type] = lab1_to_lab2_mapA(dataset[dat_type]['Label'])	
		else:
			labels[dat_type] = dataset[dat_type]['Label']

		lab_mapB = {0:0, 3:1, 4:2, 5:3, 6:4, 7:-1, 1:-1, 2:-1, 8:8, 9:9, 10:10}
		lab1_to_lab2_mapB = lambda lab_list: [lab_mapB[lab] for lab in lab_list]
		labels[dat_type] = lab1_to_lab2_mapB(labels[dat_type])

		#removing labels 9 and 10
		temp_texts = []
		temp_labels = []
		#if data_name == 'iemocap':
		for it,lab in enumerate(labels[dat_type]):
			if lab not in [-1,8,9,10]:
				temp_texts.append(texts[dat_type][it])
				temp_labels.append(labels[dat_type][it])
		texts[dat_type]=temp_texts
		labels[dat_type]=temp_labels


	'''
	save train-test split
	'''
	train_test_split_dir = "../train_test_split"

	if data_name != 'dyda_e':
		pk.dump({
		        'train_labels':labels['train'], 
		        'train_texts':texts['train'],
		        'val_labels':labels['validation'], 
		        'val_texts':texts['validation'],
		        'test_labels': labels['test'], 
		        'test_texts': texts['test']
		        }
		        , open(f"{train_test_split_dir}/"+data_name+'.pkl', 'wb'))
	 

	else:
		for s in range(4):
			lab_tr = labels['train'][s*len(labels['train'])//4 : (s+1)*len(labels['train'])//4]
			lab_val = labels['validation'][s*len(labels['validation'])//4 : (s+1)*len(labels['validation'])//4]
			lab_te = labels['test'][s*len(labels['test'])//4 : (s+1)*len(labels['test'])//4]

			txt_tr = texts['train'][s*len(texts['train'])//4 : (s+1)*len(texts['train'])//4]
			txt_val = texts['validation'][s*len(texts['validation'])//4 : (s+1)*len(texts['validation'])//4]
			txt_te = texts['test'][s*len(texts['test'])//4 : (s+1)*len(texts['test'])//4]

			print(f"Dyda-{s}")
			print(f"Number of train conversations: {len(lab_tr)}")
			print(f"Number of valid conversations: {len(lab_val)}")
			print(f"Number of test conversations: {len(lab_te)}")

			pk.dump({
			        'train_labels':lab_tr, 
			        'train_texts':txt_tr,
			        'val_labels':lab_val, 
			        'val_texts':txt_val,
			        'test_labels': lab_te, 
			        'test_texts': txt_te
			        }
			        , open(f"{train_test_split_dir}/"+data_name+f"_{s}.pkl", 'wb'))


	print(f"saved all files to the directory: {train_test_split_dir} --> {data_name}.pkl")


