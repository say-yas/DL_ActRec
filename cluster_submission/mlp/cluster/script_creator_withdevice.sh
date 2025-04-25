#!/bin/bash

# Define the range of values for num_hidden_lyr and hidden_lyr_size
num_hidden_lyr_values=(4 10 ) # 8 16 32)
hidden_lyr_size_values=(4 8 10) # 16 32)

# Define other fixed parameters
path="/home/sharareh.sayyad/HAR/deeplearning_activity_recognition/"
batch_size=10
patience=10
modelname="mlp1"

current_dir=$PWD


my_model=("mlp1" "mlp2")
for modelname in "${my_model[@]}"; do
# Loop over num_hidden_lyr and hidden_lyr_size
for num_hidden_lyr in "${num_hidden_lyr_values[@]}"; do
    for hidden_lyr_size in "${hidden_lyr_size_values[@]}"; do
        # Create a directory for the current combination
        dir_name="har_withdevice_${num_hidden_lyr}_${hidden_lyr_size}_${modelname}_batch${batch_size}_p${patience}"
        mkdir -p "$dir_name"

        cd $PWD/${dir_name}
        
        script_name="submit_har.sh"

        # Write the script line by line using echo
        echo "#!/bin/bash" > "$script_name"
        echo "" >> "$script_name"
        echo "#SBATCH --partition=class" >> "$script_name"
        echo "#SBATCH --job-name=har_withdevice" >> "$script_name"
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
        echo "time srun python /home/sharareh.sayyad/HAR/deeplearning_activity_recognition/MLP/mlp_wisdm_withdevice.py $path $num_hidden_lyr $hidden_lyr_size $batch_size $patience $modelname $PWD/" >> "$script_name"


        chmod +x "$script_name"

        

        echo "Generated script: $script_name"

        sbatch "$script_name"

        cd ${current_dir}
    done
done
done
