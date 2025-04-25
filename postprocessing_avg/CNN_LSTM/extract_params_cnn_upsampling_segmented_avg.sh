#!/bin/bash

# Define the output file
output_file="cnnlstm_nodevice_upsampling_segmented.txt"
outputsorted_file="cnnlstm_nodevice_upsampling_segmented_sorted.txt"

# Initialize the output file with headers
echo "batch_size model_type upsample lr max_length_series segment_duration num_conv_lyrs size_linear_lyr num_blocks_per_layer initial_channels lstm_hidden_size lstm_layers norm_type mean_accuracy mean_f1_score mean_gmean mean_precision mean_recall std_accuracy std_f1_score std_gmean std_precision std_recall total_params" > "$output_file"

# Find all .out files in folders starting with "har_"
find nodevice_upsampling_segmented/avg_* -type f -name "*.out" | while read -r file; do
    echo "Processing file: $file"  # Debug: Print the file being processed

    # Initialize variables
    parameters=""
    learning_rate=""
    correctness=""
    accuracy=""
    f1_score=""
    gmean=""
    precision=""
    recall=""


    # Read the file line by line
    while IFS= read -r line; do
        # echo "Processing line: $line"  # Debug: Print the line being processed

        if [[ "$line" == "parameters: "* ]]; then
            parameters="${line#parameters: }"
        elif [[ "$line" == "cnnlstmmodel:"* ]]; then
            model_type="${line#cnnlstmmodel: }"
        elif [[ "$line" == "upsample:"* ]]; then
            upsample="${line#upsample: }"
        elif [[ "$line" == "max_length_series:"* ]]; then
            max_length_series="${line#max_length_series: }"
        elif [[ "$line" == "segment_duration:"* ]]; then
            segment_duration="${line#segment_duration: }"
        elif [[ "$line" == "num_blocks_per_layer:"* ]]; then
            num_blocks_per_layer="${line#num_blocks_per_layer: }"
        elif [[ "$line" == "num_conv_lyrs:"* ]]; then
            num_conv_lyrs="${line#num_conv_lyrs: }"
        elif [[ "$line" == "initial_channels:"* ]]; then
            initial_channels="${line#initial_channels: }"
        elif [[ "$line" == "lstm_hidden_size:"* ]]; then
            lstm_hidden_size="${line#lstm_hidden_size: }"
        elif [[ "$line" == "num_lstm_layers:"* ]]; then
            lstm_layers="${line#num_lstm_layers: }"
        elif [[ "$line" == avg_acc_test:* ]]; then
            mean_std="${line#avg_acc_test: }"
            mean_acc="${mean_std%% +/- *}"
            std_acc="${mean_std##*+/- }"
        elif [[ "$line" == avg_f1score_test:* ]]; then
            mean_std="${line#avg_f1score_test: }"
            mean_f1="${mean_std%% +/- *}"
            std_f1="${mean_std##*+/- }"
        elif [[ "$line" == avg_gmean_test:* ]]; then
            mean_std="${line#avg_gmean_test: }"
            mean_gmean="${mean_std%% +/- *}"
            std_gmean="${mean_std##*+/- }"
        elif [[ "$line" == avg_precision_test:* ]]; then
            mean_std="${line#avg_precision_test: }"
            mean_prec="${mean_std%% +/- *}"
            std_prec="${mean_std##*+/- }"
        elif [[ "$line" == avg_recall_test:* ]]; then
            mean_std="${line#avg_recall_test: }"
            mean_rec="${mean_std%% +/- *}"
            std_rec="${mean_std##*+/- }"
        elif [[ "$line" == "Total params: "* ]]; then
            total_params="${line#Total params: }"
            # Remove commas from the number (optional)
            total_params="${total_params//,/}"
        fi
    done < "$file"



# Function to extract a value for a given key
extract_value() {
    local key="$1"
    echo "$parameters" | awk -F"'"$key"': " '{print $2}' | awk -F[,}] '{print $1}'
}

# Extract values
test_size=$(extract_value "test_size")
val_size=$(extract_value "val_size")
batch_size=$(extract_value "batch_size")
num_cpus=$(extract_value "num_cpus")
# modeltype=$(extract_value "modeltype")
lr=$(extract_value "lr")
num_epochs=$(extract_value "num_epochs")
verbose=$(extract_value "verbose")
num_channels=$(extract_value "num_channels")
n_classes=$(extract_value "n_classes")
patience=$(extract_value "patience")
# max_length_series=$(extract_value "max_length_series")
size_linear_lyr=$(extract_value "size_linear_lyr")
# num_conv_lyrs=$(extract_value "num_conv_lyrs")
# initial_channels=$(extract_value "initial_channels")
num_conv_lyr=$(extract_value "num_conv_lyr")
# lstm_hidden_size=$(extract_value "lstm_hidden_size")
# lstm_layers=$(extract_value "lstm_layers")
opt=$(extract_value "opt")
weights=$(echo "$parameters" | awk -F"'weights': " '{print $2}' | awk -F[}] '{print $1}')
norm_type=$(extract_value "norm_type")


    # Append the extracted data to the output file (tab-separated)
    echo "$batch_size\t$model_type\t$upsample\t$lr\t$max_length_series\t$segment_duration\t$num_conv_lyrs\t$size_linear_lyr\t$num_blocks_per_layer\t$initial_channels\t$lstm_hidden_size\t$lstm_layers\t$norm_type\t$mean_acc\t$mean_f1\t$mean_gmean\t$mean_prec\t$mean_rec\t$std_acc\t$std_f1\t$std_gmean\t$std_prec\t$std_rec\t$total_params" >> "$output_file"

    
done

echo "Data extraction complete. Results saved in $output_file."

# Sort the output file by the F1 Score column (18th column)
{
    head -n 1 "$output_file"  # Keep the header row
    tail -n +2 "$output_file" | sort -t$'\t' -k13,13nr -k14,14nr  # Sort the rest by the F1 Score column
} > "${outputsorted_file}"

echo "Sorted results saved in ${outputsorted_file}"
rm $output_file
