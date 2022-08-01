# !/bin/bash

# to run java pos tagging 

                                                                                                    
java -XX:ParallelGCThreads=2 -Xmx500m -jar ./ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar --no-confidence ./static/ark-tweet-nlp-0.3.2/unseenTweets.txt

