import findspark                                                                     
findspark.init()
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt

def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext.getOrCreate();
    sc.setLogLevel("ERROR")
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    print (counts)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    positive_count = []
    negative_count = []
    
    for c in counts:
        for word in c:
            if word[0] == "positive":
                positive_count.append(word[1])
            else:
                negative_count.append(word[1])
                
    plt.axis([-1, len(positive_count), 0, max(max(positive_count), max(negative_count))+110])
    pos,= plt.plot(positive_count, 'b-', marker='o', markersize=10)
    neg,= plt.plot(negative_count, 'g-', marker='o', markersize=10)
    plt.legend((pos,neg),('Positive', 'Negative'), loc=2)
    plt.xticks(np.arange(0, len(positive_count),1))
    plt.xlabel("Time Step")
    plt.xlabel("Word Count")
    plt.savefig("Plot")



def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    wordlist = [word for line in open(filename, 'r') for word in line.split()]
    return set(wordlist)



def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1])
    #tweets.pprint()
    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # Split each line into words
    
    
    def calc_total(newValues, total):
        if total is None:
            total = 0
        return sum(newValues, total)  # add the new values with the previous running count to get the new count


    words = tweets.flatMap(lambda line: line.split(" ")) #.filter(lambda x: (x in pwords) or (x in nwords))
    words = words.map(lambda x: ('positive', 1) if x in pwords else ('negative',1))
    wordCounts = words.reduceByKey(lambda x, y: x + y)
    total = words.updateStateByKey(calc_total)
    total.pprint()

    # Print the first ten elements of each RDD generated in this DStream to the console
    #wordCounts.pprint()
    
    
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    #counts.append(wordCounts)
    wordCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()