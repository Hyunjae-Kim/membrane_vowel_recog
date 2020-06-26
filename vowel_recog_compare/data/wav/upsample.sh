#!bash

ls *.wav | while read name; 
do
sox $name -r 100k ../wav_100k/$name
done
