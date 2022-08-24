#!/bin/bash

Task=("accuracy" "motif")
train=("cross-validation" "straightforward")
Models=("simpleConvModel") # batchNormalizationModel
Data=("enhancer")
PoolSize=(32)
Filters=(2048) #1024 
FilterSize=(12) #9
BatchSize=(1024) #1024
Optimizer=("adam")
Epochs=(20) #20





# get length of an arrays
task_len=${#Task[@]}
train_len=${#Train[@]}
model_len=${#Models[@]}
data_len=${#Data[@]}
pool_len=${#PoolSize[@]}
filter_len=${#Filters[@]}
filtersize_len=${#FilterSize[@]}
batch_len=${#BatchSize[@]}
opt_len=${#Optimizer[@]}
epoch_len=${#Epochs[@]}

# use for loop read all nameservers
mission=$((17))
counter=$((136))
for (( i=0; i<${task_len}; i++ )); do
	for (( j=0; j<${train_len}; j++ )); do
		for (( k=0; k<${model_len}; k++ )); do
			for (( l=0; l<${data_len}; l++ )); do
				for (( m=0; m<${pool_len}; m++ )); do
					for (( n=0; n<${filter_len}; n++ )); do
						for (( o=0; o<${filtersize_len}; o++ )); do
							for (( p=0; p<${batch_len}; p++ )); do
								for (( q=0; q<${opt_len}; q++ )); do
									for ((r=0; r<${epoch_len}; r++ ));do
										counter=$((counter+1))
										echo "#PBS -N enhancer$counter" > launch$counter
										echo "cd /home/u24913/thesis/code/" >> launch$counter
										echo "./time python main.py $mission ${Task[$i]} ${Train[$j]} ${Models[$k]} ${Data[$l]} ${PoolSize[$m]} ${Filters[$n]} ${FilterSize[$o]} ${BatchSize[$p]} ${Optimizer[$q]} ${Epochs[$r]} $counter" | tr "\n" " " >> launch$counter
										echo "" >> launch$counter
										qsub launch$counter
										time=`echo $line | cut -d" " -f 3 | cut -d"e" -f 1`
										cpu=`echo $line | cut -d" " -f 4 | cut -d"C" -f 1`
										mem=`echo $line | cut -d" " -f 6 | cut -d"m" -f 1`
										echo "$counter $time $cpu $mem"
										rm -r launch$counter
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

# cp /usr/bin/time .
# ./loop.sh
