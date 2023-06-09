#!/bin/bash


# Train five different teachers "GCN" "GAT" "SAGE" "MLP" "APPNP"
# on five datasets: "cora" "citeseer" "pubmed" "a-computer" "a-photo"
# Then train corresponding GLNN for each teacher

aggregated_result_file="ablation_gnn_v6.txt"
printf "Student\n" >> $aggregated_result_file    

for e in "tran" "ind"
do
    printf "%6s\n" $e >> $aggregated_result_file
    for t in "SGCN" "GIN" "GATv2" "SAGE" "GAT" "GCN"
    do
        printf "%6s\n" $t >> $aggregated_result_file
        for ds in "cora" "citeseer" "pubmed" 
        do
            printf "%10s\t" $ds >> $aggregated_result_file
            python train_student.py --exp_setting $e --teacher $t --student "SGCN" --dataset $ds --num_exp 5 \
                                    --max_epoch 200 --patience 50 >> $aggregated_result_file
        done
        printf "\n" >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file    
done
