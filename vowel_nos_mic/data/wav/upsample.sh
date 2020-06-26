#!bash

ls *.wav | while read name; 
do
sox $name -r 10k ../wav_10k/$name
done
