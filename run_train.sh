export BERT_BASE_DIR=/path/bert/uncased_L-12_H-768_A-12
export QUORA_DIR=/path/kaggle/quora/data
export CUDA_VISIBLE_DEVICES=1,0
python3 ./bert/run_classifier.py \
	  --task_name=kaggle-quora \
	    --do_train=true \
	      --do_eval=true \
	        --data_dir=$QUORA_DIR/ \
		  --vocab_file=$BERT_BASE_DIR/vocab.txt \
		    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
		      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
		        --max_seq_length=128 \
			  --train_batch_size=32 \
			    --eval_batch_size=64 \
			    --pred_batch_size=8 \
			    --learning_rate=2e-5 \
			      --num_train_epochs=3.0 \
			        --output_dir=$QUORA_DIR/result/ \
