#1
python ../../simulation_parameters.py --num_processes 100 --num_samples 60_000 --batch_size 60_000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_aedaeg  \
    --scenario stdpopsim_aedaeg;   
#2
python ../../simulation_parameters.py --num_processes 100 --num_samples 5_000 --batch_size 5_000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_anapla \
    --scenario stdpopsim_anapla;   
#3
python ../../simulation_parameters.py --num_processes 100 --num_samples 1_000 --batch_size 1_000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_anocar \
    --scenario stdpopsim_anocar;   
#4
python ../../simulation_parameters.py --num_processes 100 --num_samples 20_000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_anogam \
    --scenario stdpopsim_anogam;   
#5
##python ../../simulation_parameters.py --num_processes 100 --num_samples 1000 --batch_size 1000 \
##    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
##    --data_dir ./parameter/stdpopsim_apimel \
##    --scenario stdpopsim_apimel; 
#6
python ../../simulation_parameters.py --num_processes 100 --num_samples 100000 --batch_size 100000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_aratha \
    --scenario stdpopsim_aratha; 
#7
python ../../simulation_parameters.py --num_processes 100 --num_samples 100000 --batch_size 100000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_aratha_map \
    --scenario stdpopsim_aratha_map; 
#8
python ../../simulation_parameters.py --num_processes 100 --num_samples 200000 --batch_size 200000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_caeele \
    --scenario stdpopsim_caeele; 
#9
python ../../simulation_parameters.py --num_processes 100 --num_samples 200000 --batch_size 200000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_caeele_map \
    --scenario stdpopsim_caeele_map; 
#10
# rerun due to error in Ne
#python ../../simulation_parameters.py --num_processes 100 --num_samples 200000 --batch_size 200000 \
#    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
#    --data_dir ./parameter/stdpopsim_canfam \
#    --scenario stdpopsim_canfam; 
#11
python ../../simulation_parameters.py --num_processes 100 --num_samples 1000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_dromel \
    --scenario stdpopsim_dromel; 
#12
##python ../../simulation_parameters.py --num_processes 100 --num_samples 1000 --batch_size 1000 \
##    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
##    --data_dir ./parameter/stdpopsim_dromel_map \
##    --scenario stdpopsim_dromel_map; 
#13
python ../../simulation_parameters.py --num_processes 100 --num_samples 60000 --batch_size 60000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_drosec \
    --scenario stdpopsim_drosec; 
#14
python ../../simulation_parameters.py --num_processes 100 --num_samples 200000 --batch_size 200000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_gasacu \
    --scenario stdpopsim_gasacu; 
#15
python ../../simulation_parameters.py --num_processes 100 --num_samples 60000 --batch_size 60000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_helann \
    --scenario stdpopsim_helann; 
#16
python ../../simulation_parameters.py --num_processes 100 --num_samples 1000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_helmel \
    --scenario stdpopsim_helmel; 


