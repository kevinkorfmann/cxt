
python simulation.py \
    --num_processes 100 \
    --num_samples 25_000 \
    --batch_size 1000 \
    --start_batch 0 \
    --pivot_A 0 \
    --pivot_B 1 \
    --data_dir /sietch_colab/kkor/llm  \
    --scenario llm_ne_sawtooth   \
    --randomize_pivots True;

################################ no used
#python simulation.py \
#    --num_processes 100 \
#    --num_samples 25_000 \
#    --batch_size 1000 \
#    --start_batch 0 \
#    --pivot_A 0 \
#    --pivot_B 1 \
#    --data_dir /sietch_colab/kkor/dataset/llm  \
#    --scenario llm_ne_constant_2   \
#    --randomize_pivots True;

#python simulation.py \
#    --num_processes 100 \
#    --num_samples 25_000 \
#    --batch_size 1000 \
#    --start_batch 0 \
#    --pivot_A 0 \
#    --pivot_B 1 \
#    --data_dir /sietch_colab/kkor/dataset/llm  \
#    --scenario llm_ne_constant_3   \
#    --randomize_pivots True;

#python simulation.py \
#    --num_processes 100 \
#    --num_samples 25_000 \
#    --batch_size 1000 \
#    --start_batch 0 \
#    --pivot_A 0 \
#    --pivot_B 1 \
#    --data_dir /sietch_colab/kkor/dataset/llm  \
#    --scenario llm_ne_constant_4   \
#    --randomize_pivots True;


#python simulation.py \
#    --num_processes 100 \
#    --num_samples 25_000 \
#    --batch_size 1000 \
#    --start_batch 0 \
#    --pivot_A 0 \
#    --pivot_B 1 \
#    --data_dir /sietch_colab/kkor/dataset/llm  \
#    --scenario llm_ne_constant_5   \
#    --randomize_pivots True;
################################ no used



python simulation.py \
    --num_processes 100 \
    --num_samples 10_000 \
    --batch_size 1000 \
    --start_batch 0 \
    --pivot_A 0 \
    --pivot_B 1 \
    --data_dir /sietch_colab/kkor/llm  \
    --scenario llm_hard_sweeps   \
    --randomize_pivots True;

python simulation.py \
    --num_processes 75 \
    --num_samples 10_000 \
    --batch_size 1000 \
    --start_batch 0 \
    --pivot_A 0 \
    --pivot_B 1 \
    --data_dir /sietch_colab/kkor/llm  \
    --scenario llm_island_3pop   \
    --randomize_pivots True;

################################ no used
#python simulation.py \
#    --num_processes 100 \
#    --num_samples 10_000 \
#    --batch_size 1000 \
#    --start_batch 0 \
#    --pivot_A 0 \
#    --pivot_B 1 \
#    --data_dir /sietch_colab/kkor/llm  \
#    --scenario llm_island_5pop   \
#    --randomize_pivots True;
################################ no used




python simulation.py \
    --num_processes 100 \
    --num_samples 100_000 \
    --batch_size 1000 \
    --start_batch 0 \
    --pivot_A 0 \
    --pivot_B 1 \
    --data_dir /sietch_colab/kkor/llm  \
    --scenario llm_ne_constant   \
    --randomize_pivots True
