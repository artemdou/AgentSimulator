python simulate.py --log_path raw_data/LoanApp.csv.gz --case_id case_id --activity_name activity --resource_name resource --end_timestamp end_time --start_timestamp start_time
python simulate.py --log_path raw_data/P2P.csv --case_id case:concept:name --activity_name concept:name --resource_name Resource --end_timestamp time:timestamp --start_timestamp start_timestamp 
python simulate.py --log_path raw_data/Production.csv --case_id caseid --activity_name task --resource_name user --end_timestamp end_timestamp --start_timestamp start_timestamp 
python simulate.py --log_path raw_data/ConsultaDataMining.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp 
python simulate.py --log_path raw_data/BPIC_2012_W.csv --case_id case:concept:name --activity_name Activity --resource_name Resource --end_timestamp time:timestamp --start_timestamp start_timestamp 
python simulate.py --log_path raw_data/cvs_pharmacy.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp 
python simulate.py --log_path raw_data/BPIC_2017_W.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp
python simulate.py --log_path raw_data/Confidential_1000.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp 
python simulate.py --log_path raw_data/Confidential_2000.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp 
python optimize_pro.py \      
  --log_path raw_data/LoanApp.csv.gz \
  --case_id case_id \
  --activity_name activity \
  --resource_name resource \
  --start_timestamp start_time \
  --end_timestamp end_time \
  --pop_size 50 \
  --num_gen 30 \
  --runs_per_fitness 15 \
  --n_cores 7 \
  --costs_path costs.json \
  --objectives "cost,time,agents_per_case"
python simulate.py --log_path raw_data/LoanApp.csv.gz --case_id case_id --activity_name activity --resource_name resource --end_timestamp end_time --start_timestamp start_time --execution_type greedy --weights '{"progress": 0.5, "task_cost": 0.5}'