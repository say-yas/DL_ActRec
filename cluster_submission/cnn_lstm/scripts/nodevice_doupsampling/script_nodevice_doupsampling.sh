#!/bin/bash

# Define the range of values for nn model
num_conv_lyrs_values=(2 6 8) # 8 16 32)
size_linear_lyr_values=(8 16) # 10) # 16 32)
hidden_lyr_size_values=(8) # 16 32)
num_blocks_per_layer_values=(2 4)
initial_channels_values=(8) # 16 32 64)
lstm_hidden_sizes=(4 8)
lstm_layers=(2)

# Define other fixed parameters
path="/home/sharareh.sayyad/HAR/deeplearning_activity_recognition/"
batch_size=64
patience=20
norm_type="per-timestep" # "per-timestep" "per-channel"

my_models=("cnnlstm1" "cnnlstmskip" "cnnlstmbatchnorm" "cnnlstmbatchnormparallel" "cnnlstmskipparallel")
lrs=(1e-3)
max_length_timesteps=(2000 4000)
doupsamplings=("SMOTEENN" "SMOTETomek")


current_dir=$PWD

for lstm_layer in "${lstm_layers[@]}"; do
for lstm_hidden_size in "${lstm_hidden_sizes[@]}"; do
for doupsampling in "${doupsamplings[@]}"; do
for max_length_timestep in "${max_length_timesteps[@]}"; do
for lr in "${lrs[@]}"; do
for modelname in "${my_models[@]}"; do
for num_conv_lyrs in "${num_conv_lyrs_values[@]}"; do
for size_linear_lyr in "${size_linear_lyr_values[@]}"; do
    for hidden_lyr_size in "${hidden_lyr_size_values[@]}"; do
    for num_blocks_per_layer in "${num_blocks_per_layer_values[@]}"; do
    for initial_channels in "${initial_channels_values[@]}"; do

        # Create a directory for the current combination
        dir_name="nodevice_${modelname}_batch${batch_size}_p${patience}_norm${norm_type}_maxt${max_length_timestep}_conv${num_conv_lyrs}_lnr${size_linear_lyr}_lstm${lstm_layer}_lstmsize${lstm_hidden_size}_hdlyrsz${hidden_lyr_size}_nblocks${num_blocks_per_layer}_inich${initial_channels}_lr${lr}_doupsampling${doupsampling}"
        mkdir -p "$dir_name"

        cd $PWD/${dir_name}
        
        script_name="dolstmcnn.sh"

        # Write the script line by line using echo
        echo "#!/bin/bash" > "$script_name"
        echo "" >> "$script_name"
        echo "#SBATCH --partition=class" >> "$script_name"
        echo "#SBATCH --job-name=wdev" >> "$script_name"
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

         
        echo "time srun python /home/sharareh.sayyad/HAR/deeplearning_activity_recognition/CNN_LSTM/cnn_wisdm_nodevice_withdownupsampling.py $path $max_length_timestep $num_conv_lyrs $size_linear_lyr $num_blocks_per_layer $initial_channels $lstm_hidden_size $lstm_layer $batch_size $patience $modelname $norm_type $lr $doupsampling $PWD/" >> "$script_name"


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

