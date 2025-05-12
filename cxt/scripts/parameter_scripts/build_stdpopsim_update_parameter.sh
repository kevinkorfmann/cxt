python ../../simulation_parameters.py --num_processes 100 --num_samples 100000 --batch_size 100000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_musmus \
    --scenario stdpopsim_musmus; 

python ../../simulation_parameters.py --num_processes 100 --num_samples 100000 --batch_size 100000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_ratnor \
    --scenario stdpopsim_musmus; 

python ../../simulation_parameters.py --num_processes 100 --num_samples 100000 --batch_size 100000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_gorgor \
    --scenario stdpopsim_gorgor; 

python ../../simulation_parameters.py --num_processes 50 --num_samples 100000 --batch_size 100000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_orysat \
    --scenario stdpopsim_orysat; 

python ../../simulation_parameters.py --num_processes 100 --num_samples 100000 --batch_size 100000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_susscr \
    --scenario stdpopsim_susscr; 

python ../../simulation_parameters.py --num_processes 100 --num_samples 100000 --batch_size 100000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir ./parameter/stdpopsim_phosin \
    --scenario stdpopsim_phosin; 


