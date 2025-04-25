#!/bin/bash

# Define the range of values for nn model
embed_size_values=(8 16 32)
nhead_values=(4 8 16)
dim_feedforward_values=(16 32 64) 
num_encoderlayers_values=(1 2 4)            
dropout=0.0
conv1d_emb_types=("True" "False") 
conv1d_kernel_size_values=(3 5 7) 
size_linear_layers_values=(16 32 64)

# Define other fixed parameters
path="/home/sharareh.sayyad/HAR/deeplearning_activity_recognition/"
batch_size=32
patience=20
norm_type="per-channel" # "per-timestep" "per-channel"

my_models=("trans1")
lrs=(1e-3)
max_length_timesteps=(4000)
segmented_durations=(50 100)
doupsamplings=("SMOTEENN" "SMOTETomek")


current_dir=$PWD

for segmented_duration in "${segmented_durations[@]}"; do
for doupsampling in "${doupsamplings[@]}"; do
for max_length_timestep in "${max_length_timesteps[@]}"; do
for lr in "${lrs[@]}"; do
for modelname in "${my_models[@]}"; do
for conv1d_emb in "${conv1d_emb_types[@]}"; do
for embed_size in "${embed_size_values[@]}"; do
for nhead in "${nhead_values[@]}"; do
for dim_feedforward in "${dim_feedforward_values[@]}"; do
for conv1d_kernel_size in "${conv1d_kernel_size_values[@]}"; do
for size_linear_layers in "${size_linear_layers_values[@]}"; do
for num_encoderlayers in "${num_encoderlayers_values[@]}"; do


    # Create a directory for the current combination
    dir_name="nodevice_${modelname}_batch${batch_size}_p${patience}_norm${norm_type}_maxt${max_length_timestep}_segt${segmented_duration}_lr${lr}_doupsampling${doupsampling}_embed${embed_size}_nhead${nhead}_dimfeed${dim_feedforward}_conv1demb${conv1d_emb}_conv1dkernel${conv1d_kernel_size}_size_linear_lyrs${size_linear_layers}_num_encoderlayers${num_encoderlayers}"
    mkdir -p "$dir_name"

    cd $PWD/${dir_name}
    
    script_name="segdocnnlstm.sh"

    # Write the script line by line using echo
    echo "#!/bin/bash" > "$script_name"
    echo "" >> "$script_name"
    echo "#SBATCH --partition=class" >> "$script_name"
    echo "#SBATCH --job-name=doupseg" >> "$script_name"
    echo "#SBATCH --output=%x_%j.out" >> "$script_name"
    echo "#SBATCH --error=%x_%j.err" >> "$script_name"
    echo "#SBATCH --mail-type=ALL" >> "$script_name"
    echo "#SBATCH --time=7-00:00:00" >> "$script_name"
    echo "#SBATCH --nodes=1" >> "$script_name"
    echo "#SBATCH --ntasks-per-node=1" >> "$script_name"
    echo "#SBATCH --cpus-per-task=30" >> "$script_name"
    echo "#SBATCH --gres=gpu:tesla:1" >> "$script_name"
    echo "" >> "$script_name"
    echo "module load anaconda3" >> "$script_name"
    echo "conda init bash" >> "$script_name"
    echo "conda activate har" >> "$script_name"
    echo "" >> "$script_name"

    echo "time srun python /home/sharareh.sayyad/HAR/deeplearning_activity_recognition/Transformer/transformer_wisdm_nodevice_withdownupsampling_segmented.py $path $max_length_timestep $modelname $embed_size $nhead $dim_feedforward $num_encoderlayers $conv1d_kernel_size $size_linear_layers $conv1d_emb $dropout $batch_size $patience $norm_type $lr $segmented_duration  $doupsampling $PWD/" >> "$script_name"

    chmod +x "$script_name"

    echo "Generated script: $script_name"

    sbatch "$script_name"

    cd ${current_dir}
done
done
done
done
done
done
done
done
done
done
done
done
