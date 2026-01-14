#!/bin/bash

# Configuration
DEVICE="default"        # PipeWire manages this
DURATION=10             # Total seconds to run
SAMPLES_PER_SEC=44100   # Standard sample rate

echo "Time | Amplitude (RMS)"
echo "----------------------"

for (( i=1; i<=$DURATION; i++ ))
do
    # 1. arecord captures 1 second of 16-bit Mono RAW audio
    # 2. 'od' converts the raw binary bytes into signed decimal integers
    # 3. 'awk' squares each sample, averages them, and takes the square root
    RMS=$(arecord -D $DEVICE -d 1 -f S16_LE -c1 -r $SAMPLES_PER_SEC -t raw 2>/dev/null | \
          od -An -t d2 | \
          awk '{ for(i=1; i<=NF; i++) { sum += $i*$i; n++ } } 
               END { if (n>0) print sqrt(sum/n); else print 0 }')

    echo " ${i}s  | $RMS"
done

echo "----------------------"
