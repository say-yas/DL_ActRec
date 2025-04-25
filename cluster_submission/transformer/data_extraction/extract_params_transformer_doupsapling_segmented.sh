#!/bin/bash

# Define the output file
output_file="transformer_nodevice.txt"
outputsorted_file="transformer_nodevice_sorted.txt"

# Initialize the output file with headers
echo "batch_size transformermodel doupsample lr max_length_series segment_duration nhead_encoder conv1d_kernel_size bool_conv1d_emb embed_size dim_feedforward norm_type num_encoderlayers size_linear_layers accuracy f1_score gmean precision recall" > "$output_file"

# Find all .out files in folders starting with "har_"
find nodevice_doupsampling/nodevice_* -type f -name "*.out" | while read -r file; do
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
        elif [[ "$line" == "transformermodel:"* ]]; then
            transformermodel="${line#transformermodel: }"
        elif [[ "$line" == "doupsample:"* ]]; then
            doupsample="${line#doupsample: }"
        elif [[ "$line" == "bool_conv1d_emb:"* ]]; then
            bool_conv1d_emb="${line#bool_conv1d_emb: }"
        elif [[ "$line" == "max_length_series:"* ]]; then
            max_length_series="${line#max_length_series: }"
        elif [[ "$line" == "segment_duration:"* ]]; then
            segment_duration="${line#segment_duration: }"
        elif [[ "$line" == "embed_size:"* ]]; then
            embed_size="${line#embed_size: }"
        elif [[ "$line" == "nhead_encoder:"* ]]; then
            nhead_encoder="${line#nhead_encoder: }"
        elif [[ "$line" == "dim_feedforward:"* ]]; then
            dim_feedforward="${line#dim_feedforward: }"
        elif [[ "$line" == "Accuracy on test set: "* ]]; then
            accuracy="${line#Accuracy on test set: }"
        elif [[ "$line" == "F1score on test set: "* ]]; then
            f1_score="${line#F1score on test set: }"
        elif [[ "$line" == "Gmean on test set: "* ]]; then
            gmean="${line#Gmean on test set: }"
        elif [[ "$line" == "Precision on test set: "* ]]; then
            precision="${line#Precision on test set: }"
        elif [[ "$line" == "Recall on test set: "* ]]; then
            recall="${line#Recall on test set: }"
        fi
    done < "$file"



# Function to extract a value for a given key
extract_value() {
    local key="$1"
    echo "$parameters" | awk -F"'"$key"': " '{print $2}' | awk -F[,}] '{print $1}'
}

# Extract values
# segment_duration=$(extract_value "segment_duration")
test_size=$(extract_value "test_size")
val_size=$(extract_value "val_size")
batch_size=$(extract_value "batch_size")
num_cpus=$(extract_value "num_cpus")
lr=$(extract_value "lr")
num_epochs=$(extract_value "num_epochs")
verbose=$(extract_value "verbose")
num_encoderlayers=$(extract_value "num_encoderlayers")
n_classes=$(extract_value "n_classes")
patience=$(extract_value "patience")
# max_length_series=$(extract_value "max_length_series")
conv1d_kernel_size=$(extract_value "conv1d_kernel_size")
size_linear_layers=$(extract_value "size_linear_layers")
opt=$(extract_value "opt")
weights=$(echo "$parameters" | awk -F"'weights': " '{print $2}' | awk -F[}] '{print $1}')
norm_type=$(extract_value "norm_type")


    # Append the extracted data to the output file (tab-separated)
    echo "$batch_size\t$transformermodel\t$doupsample\t$lr\t$max_length_series\t$segment_duration\t$nhead_encoder\t$conv1d_kernel_size\t$bool_conv1d_emb\t$embed_size\t$dim_feedforward\t$norm_type\t$num_encoderlayers\t$size_linear_layers\t$accuracy\t$f1_score\t$gmean\t$precision\t$recall" >> "$output_file"

    
done

echo "Data extraction complete. Results saved in $output_file."

# Sort the output file by the F1 Score column (18th column)
{
    head -n 1 "$output_file"  # Keep the header row
    tail -n +2 "$output_file" | sort -t$'\t' -k14,14nr -k15,15nr  # Sort the rest by the F1 Score column
} > "${outputsorted_file}"

echo "Sorted results saved in ${outputsorted_file}"
