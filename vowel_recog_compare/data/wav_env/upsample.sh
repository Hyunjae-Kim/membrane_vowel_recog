#!bash

ls *.wav | while read name; 
do
sox $name -r 1000k ../wav_1000k_env/$name
done
