@echo off

                                                                                                    
java -XX:ParallelGCThreads=2 -Xmx500m -jar ./ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar --no-confidence ./static/ark-tweet-nlp-0.3.2/unseenTweets.txt

:: C:\Users\HP\Desktop\fypAPI_local\static\ark-tweet-nlp-0.3.2\ark-tweet-nlp-0.3.2.jar
