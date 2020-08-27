import org.bytedeco.opencv.presets.opencv_core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class HMM {
    Map<String, Map<String, Double>> observationProb;  //my observation map
    Map<String, Map<String, Double>> transitionProb;       // my transition map
    Map<String, Double> stateProb;                          // my map to keep account of the probability of the first tags that can appear
    String lines, otherline;                                // strings for reading file
    BufferedReader tagFile, sentenceFile;                   // bufferReaders for opening file
    int total;                                              // integer for total count

    public HMM( String tagFiles, String sentenceFiles ) { // the constructor that creates the transitionProb and observationProb maps
        try {
            tagFile = new BufferedReader(new FileReader(tagFiles));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        try {
            sentenceFile = new BufferedReader(new FileReader(sentenceFiles));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        try {
            transitionProb = new HashMap<>();
            observationProb = new HashMap<>();
            stateProb = new HashMap<>();

            assert tagFile != null;
            assert sentenceFile != null;
            while (((lines = tagFile.readLine()) != null) && ((otherline = sentenceFile.readLine()) != null)) {
                String[] otherPieces = otherline.split(" "); //read one line from each file
                String[] pieces = lines.split(" ");
                String prev = null;
                for (int v = 0; v < pieces.length; v++) {

                    if (!stateProb.containsKey(pieces[v]) && v == 0 ) {   //fine the probability and establish the stateProb map
                        stateProb.put(pieces[v], 1.0);
                    }
                    else if(v == 0) {
                        stateProb.put(pieces[v], stateProb.get(pieces[v]) + 1);
                    }

                    if (!transitionProb.containsKey(pieces[v])) {  // iterate and create the transitionMap and initalize using the tags
                        transitionProb.put(pieces[v], new HashMap<>());
                        if (!observationProb.containsKey(pieces[v])) {
                            observationProb.put(pieces[v], new HashMap<>());
                        }
                    }
                    if (prev != null) {  // if there is a change in prev and curr then increment transition number
                        if (!transitionProb.get(prev).containsKey(pieces[v])) {
                            transitionProb.get(prev).put(pieces[v], 1.0); // if it doesn't contain tag then create one and add 1 to transition number
                        }
                        else {
                            transitionProb.get(prev).put(pieces[v], (transitionProb.get(prev).get(pieces[v])) + 1);  // if tag already exist then increment by 1
                        }
                    }
                    prev = pieces[v];  //update prev afterwards

                    if (!observationProb.get(pieces[v]).containsKey(otherPieces[v])) {  // create the observation inside maps
                        observationProb.get(pieces[v]).put(otherPieces[v], 0.0);  // if observation found first time, initialise and start with 1
                    }
                    observationProb.get(pieces[v]).put(otherPieces[v], observationProb.get(pieces[v]).get(otherPieces[v]) + 1); // if found not first time, increment by 1
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (String x : transitionProb.keySet()) {  // re write the scores for transition using natural log
            total = 0;
            for (String y : transitionProb.get(x).keySet()) { // iterate through keys and keep total count of keys
                total += transitionProb.get(x).get(y);
            }

            for (String z : transitionProb.get(x).keySet()) { // use log and total and set the new score
                Double d = Math.log(transitionProb.get(x).get(z) / total);
                transitionProb.get(x).put(z, d);
            }
        }

        for (String e : observationProb.keySet()) { // doing the same thing we did for transitionProb
            total = 0;
            for (String b : observationProb.get(e).keySet()) {
                total += observationProb.get(e).get(b);
            }

            for (String f : observationProb.get(e).keySet()) {
                Double d = Math.log(observationProb.get(e).get(f) / total);
                observationProb.get(e).put(f, d);
            }
        }
        total = 0;
        for (String q : stateProb.keySet()) { //doing the same for stateProb
            total += stateProb.get(q);
        }
        for (String q : stateProb.keySet()) {
            Double d = Math.log(stateProb.get(q) / total);
            stateProb.put(q, d);
        }

        try {
            tagFile.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            sentenceFile.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public ArrayList<String> viterbi(String ss) {
        ArrayList<String> path = new ArrayList<>();
        String[] pieces = ss.split(" ");
        ArrayList<Map<String, Map<String, String>>> backTrace = new ArrayList<>();  //create my backTrace Map that keeps account of best previous path
        ArrayList<Map<String, Map<String, Double>>> backTraceProb = new ArrayList<>(); // create my backTraceProb Map that keeps account of best previous path's probability
        double d = -100;

        if (pieces.length > 0) {
            Map<String, Map<String, String>> temp = new HashMap<>(); //create the maps for the first observations
            Map<String, Map<String, Double>> temp1 = new HashMap<>();
            temp.put(pieces[0], new HashMap<>());
            temp1.put(pieces[0], new HashMap<>());
            for (String s : observationProb.keySet()) { //iterate through observations and find the key, and add its tag and probability to its map
                if (observationProb.get(s).containsKey(pieces[0])) {
                    d = observationProb.get(s).get(pieces[0]);
                    if(stateProb.get(s) != null){
                        d += stateProb.get(s);
                    }
                    temp.get(pieces[0]).put(s, null);
                    temp1.get(pieces[0]).put(s, d);
                }
            }

            if( d == -100 ){ // if word not found in observations, add the probable tags with their probability to the backtrace and backtraceProb
                for(String s: stateProb.keySet()){
                    d = stateProb.get(s);
                    temp.get(pieces[0]).put(s, null);
                    temp1.get(pieces[0]).put(s, d);
                }
            }
            backTrace.add(temp);
            backTraceProb.add(temp1);
        }

        for (int i = 1; i < pieces.length; i++) {  //for the rest of the words
            double currMax = -100;
            String currMaxStr = null;

            Map<String, Map<String, String>> newMap = new HashMap<>(); //create their maps to be added to the tracer maps
            newMap.put(pieces[i], new HashMap<>());

            Map<String, Map<String, Double>> newMap1 = new HashMap<>();
            newMap1.put(pieces[i], new HashMap<>());

            Map<String, Map<String, String>> temp = backTrace.get(i - 1); //find the previous index maps to find the best path
            Map<String, Map<String, Double>> temp1 = backTraceProb.get(i - 1);
            for (String s : observationProb.keySet()) {  //iterate through observations and find the tags
                if (observationProb.get(s).containsKey(pieces[i])) {
                    for (String v : temp.get(pieces[i - 1]).keySet()) {
                        if (transitionProb.get(v).containsKey(s)) {
                            d = temp1.get(pieces[i - 1]).get(v) + transitionProb.get(v).get(s) + observationProb.get(s).get(pieces[i]); //compare best path using transitions from the
                            if (d > currMax) {                                                                                          //tags in previous index
                                currMax = d;
                                currMaxStr = v;
                            }
                        }
                        else if(!s.equals(v)){ //if no transition found
                            d = -90; //set a lesser score and just use previous tag
                            if (d > currMax){
                                currMax = d;
                                currMaxStr = v;
                            }
                        }
                    }
                    if(currMaxStr != null) { //then add that to our map for that word
                        newMap.get(pieces[i]).put(s, currMaxStr);
                        newMap1.get(pieces[i]).put(s, currMax);
                    }
                }
            }
            if ( currMax == -100 && currMaxStr == null){ //if word not in file
                for (String v: temp.get(pieces[i-1]).keySet()){
                    for (String u: transitionProb.get(v).keySet()){  //find the likeliest previous tags
                        if(!u.equals(".")) {
                            d = transitionProb.get(v).get(u); //compare their score and choose the best tag
                            if (d > currMax) {
                                currMax = d;
                                currMaxStr = u;
                            }
                        }
                    }
                    newMap.get(pieces[i]).put(currMaxStr, v); //use that to add to our map for that word
                    newMap1.get(pieces[i]).put(currMaxStr, currMax);
                }
            }
            backTrace.add(newMap);  //add that map for that word with its best previous path tp our tracers
            backTraceProb.add(newMap1);
        }

        Map<String, Map<String, Double>> trial = backTraceProb.get(backTraceProb.size() - 1); //find the last index from our tracer to backtrack and find the pattern

        String currMaxStr = null;
        double currMax = -100;
        for (String s : trial.get(pieces[pieces.length - 1]).keySet()) { //find the current pattern and prob score
            if (trial.get(pieces[pieces.length - 1]).get(s) > currMax) {
                currMax = trial.get(pieces[pieces.length - 1]).get(s);
                currMaxStr = s;
            }
        }
        String curr = currMaxStr;
        for (int i = pieces.length - 1; i >= 0; i--) { //backtrack through the series of maps and add to our list to return
            if (curr != null) {
                path.add(0, curr);
                curr = backTrace.get(i).get(pieces[i]).get(curr);
            }
        }
        return path;
    }

    public double performanceTest(String trialFiles, String testFiles ){ //take two files to compare performance of our viterbi
        double correct = 0;
        double incorrect = 0;
        BufferedReader trialFile = null, testFile = null;

        try {
             trialFile = new BufferedReader(new FileReader(trialFiles)); //open both files
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        try{
            testFile = new BufferedReader( new FileReader(testFiles));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String line, line1;
        try{
            assert trialFile != null;
            assert testFile != null;
            while((line = trialFile.readLine()) != null && (line1 = testFile.readLine()) != null){ //read a line

                ArrayList<String> trialList = viterbi(line); //viterbi the trial file
                String[] testList = line1.split(" ");

                for( int i = 0; i < trialList.size(); i ++){ //then compare index similarity to test file
                    if(trialList.get(i).equals(testList[i])){
                        correct += 1;  //increment accordingly
                    }
                    else{
                        incorrect += 1;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return (correct/(correct+incorrect))*100;  //return percentage
    }

    public ArrayList<String> scanner(){  //create a scanner for user to input sentence and apply viterbi and return pattern
        Scanner userInput = new Scanner(System.in);

        while(true) {
            System.out.println("write out your sentence");
            System.out.println(viterbi(userInput.nextLine()));
        }

    }

    public static void main(String[] args) {
        BufferedReader file = null;
        HMM sudi = new HMM("simple-train-tags.txt", "simple-train-sentences.txt" );  //test on simples test files
        System.out.println(sudi.performanceTest("simple-test-sentences.txt", "simple-test-tags.txt"));

        HMM sudi1 = new HMM("brown-train-tags.txt", "brown-train-sentences.txt");   //test on brown files
        System.out.println(sudi1.performanceTest("brown-test-sentences.txt", "brown-test-tags.txt"));
        System.out.println(sudi1.scanner());
    }
}