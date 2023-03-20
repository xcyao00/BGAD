# Train MvtecAD
# seed = 0, img_auc = 0.994, pixel_auc = 0.993
python main.py --flow_arch conditional_flow_model --gpu 0 --data_path /path/to/your/dataset --with_fas --data_strategy 0,1 --num_anomalies 10 --not_in_test --exp_name bgad_fas_10 --focal_weighting --pos_beta 0.01 --margin_tau 0.1

# Train BTAD
python main.py --flow_arch conditional_flow_model --gpu 0 --dataset btad --data_path /path/to/your/dataset --with_fas --data_strategy 0,1 --num_anomalies 10 --not_in_test --exp_name bgad_fas_10 --focal_weighting --pos_beta 0.01 --margin_tau 0.1

# Test MvtecAD
python test.py --flow_arch conditional_flow_model --gpu 0 --data_path /path/to/your/dataset --checkpoint /path/to/output/dir --phase test --pro 
# Test BTAD
python test.py --flow_arch conditional_flow_model --gpu 0 --dataset btad --data_path /path/to/your/dataset --checkpoint /path/to/output/dir --phase test --pro