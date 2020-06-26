#!bash

ls *.wav | while read name; 
do
sox $name -r 1000k water/$name
done
