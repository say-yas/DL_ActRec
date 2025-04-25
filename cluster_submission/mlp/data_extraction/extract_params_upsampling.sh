#!/bin/bash

# Define the output file
output_file="output_table_upsampling.txt"
outputsorted_file="output_table_upsampling_sorted.txt"

# Initialize the output file with headers
echo "Test-Size Val-Size Batch-Size modeltype lr NFeatures NClasses NumHiddenLayers HiddenLayerSize Optimizer upsampling Accuracy F1Score" > "$output_file"

# Find all .out files in folders starting with "har_"
find withoutdevice_upsampling/har_* -type f -name "*.out" | while read -r file; do
    echo "Processing file: $file"  # Debug: Print the file being processed

    # Initialize variables
    parameters=""
    learning_rate=""
    correctness=""
    accuracy=""
    f1_score=""

    # Read the file line by line
    while IFS= read -r line; do
        # echo "Processing line: $line"  # Debug: Print the line being processed

        if [[ "$line" == "parameters: "* ]]; then
            parameters="${line#parameters: }"
            # echo "Found parameters: $parameters"  

        elif [[ "$line" == *"upsampling:"* ]]; then
            # Extract model type from the line
            upsampling=$(echo "$line" | awk -F"upsampling: " '{print $2}' | awk '{print $NF}')
            # echo "Found model_type: $model_type"  # Debug: Print extracted model type

        # elif [[ "$line" == "Learning Rate: "* ]]; then
        #     learning_rate="${line#Learning Rate: }"
        #     echo "Found learning_rate: $learning_rate" 
#        elif [[ "$line" == "correctness: "* ]]; then
#            correctness="${line#correctness: }"
            # echo "Found correctness: $correctness"  
        elif [[ "$line" == "Accuracy on test set: "* ]]; then
            accuracy="${line#Accuracy on test set: }"
            # echo "Found accuracy: $accuracy"  
        elif [[ "$line" == "F1score on test set: "* ]]; then
            f1_score="${line#F1score on test set: }"
            # echo "Found f1_score: $f1_score"  
        fi
    done < "$file"


    # Extract individual parameters from the JSON-like string using awk
    test_size=$(echo "$parameters" | awk -F"'test_size': " '{print $2}' | awk -F[,}] '{print $1}')
    val_size=$(echo "$parameters" | awk -F"'val_size': " '{print $2}' | awk -F[,}] '{print $1}')
    batch_size=$(echo "$parameters" | awk -F"'batch_size': " '{print $2}' | awk -F[,}] '{print $1}')
    model_type=$(echo "$parameters" | awk -F"'modeltype': " '{print $2}' | awk -F[,}] '{print $1}')
    lr=$(echo "$parameters" | awk -F"'lr': " '{print $2}' | awk -F[,}] '{print $1}')
    # num_epochs=$(echo "$parameters" | awk -F"'num_epochs': " '{print $2}' | awk -F[,}] '{print $1}')
    # verbose=$(echo "$parameters" | awk -F"'verbose': " '{print $2}' | awk -F[,}] '{print $1}')
    n_features=$(echo "$parameters" | awk -F"'n_features': " '{print $2}' | awk -F[,}] '{print $1}')
    n_classes=$(echo "$parameters" | awk -F"'n_classes': " '{print $2}' | awk -F[,}] '{print $1}')
    # patience=$(echo "$parameters" | awk -F"'patience': " '{print $2}' | awk -F[,}] '{print $1}')
    num_hidden_lyr=$(echo "$parameters" | awk -F"'num_hidden_lyr': " '{print $2}' | awk -F[,}] '{print $1}')
    hidden_lyr_size=$(echo "$parameters" | awk -F"'hidden_lyr_size': " '{print $2}' | awk -F[,}] '{print $1}')
    opt=$(echo "$parameters" | awk -F"'opt': '" '{print $2}' | awk -F"'" '{print $1}')
    # weights=$(echo "$parameters" | awk -F"'weights': tensor(" '{print $2}' | awk -F")" '{print $1}')


    # Append the extracted data to the output file (tab-separated)
    echo "$test_size\t$val_size\t$batch_size\t$model_type\t$lr\t$n_features\t$n_classes\t$num_hidden_lyr\t$hidden_lyr_size\t$opt\t$upsampling\t$accuracy\t$f1_score" >> "$output_file"

    
done

echo "Data extraction complete. Results saved in $output_file."

# Sort the output file by the F1 Score column (18th column)
{
    head -n 1 "$output_file"  # Keep the header row
    tail -n +2 "$output_file" | sort -t$'\t' -k12,12nr -k13,13nr  # Sort the rest by the F1 Score column
} > "${outputsorted_file}"

echo "Sorted results saved in ${outputsorted_file}"
