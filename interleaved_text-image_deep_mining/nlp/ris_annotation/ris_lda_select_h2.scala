import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

import java.io._;

object ris_lda_select{
  def main(args: Array[String]) = {

    for (h2tpi <- args(1).toInt to args(2).toInt){

      println(h2tpi)

      val source = CSVFile("../data/ris_reports_per_topic_h2/" + 
        "ris_reports_per_h1_topic_" + h2tpi + ".csv") ~> IDColumn(1);

      val tokenizer = {
        SimpleEnglishTokenizer() ~>            // tokenize on space and punctuation
        CaseFolder() ~>                        // lowercase everything
        WordsAndNumbersOnlyFilter() ~>         // ignore non-words and non-numbers
        MinimumLengthFilter(3)                 // take terms with >=3 characters
      }

      val text = {
        source ~>                              // read from the source file
        Column(3) ~>                           // select column containing text
        TokenizeWith(tokenizer) ~>             // tokenize with tokenizer above
        TermCounter() ~>                       // collect counts (needed below)
        TermMinimumDocumentCountFilter(3) ~>   // filter terms in <4 docs
        TermStopListFilter(List("0001","0002","Report")) ~>
        TermDynamicStopListFilter(10) ~>       // filter out 30 most common terms
        DocumentMinimumLengthFilter(5)         // take only docs with >=5 terms
      }

      // set aside 80 percent of the input text as training data ...
      val numTrain = text.data.size * 4 / 5;

      // build a training dataset
      val training = LDADataset(text ~> Take(numTrain));
       
      // build a test dataset, using term index from the training dataset 
      val testing  = LDADataset(text ~> Drop(numTrain));

      // a list of pairs of (number of topics, perplexity)
      var scores = List.empty[(Int,Double)];


      // loop over various numbers of topics, training and evaluating each model
      for (numTopics <- List(10,20,30,40,50,60,70,80,90,100)) {
        val params = LDAModelParams(numTopics = numTopics, dataset = training);
        val output = file("lda-"+training.signature+"-"+params.signature);
        val model = TrainCVB0LDA(params, training, output=null, maxIterations=5000);
        
        println("[perplexity] computing at "+numTopics);

        val perplexity = model.computePerplexity(testing);
        
        println("[perplexity] perplexity at "+numTopics+" topics: "+perplexity);

        scores :+= (numTopics, perplexity);
      }

      val writer = new PrintWriter(new File("../data/ris_reports_per_topic_h2/" +
        "ris_reports_per_h1_topic_" + h2tpi + "_sel" + ".txt" ))
      for ((numTopics,perplexity) <- scores) {
        writer.write("[perplexity] perplexity at "+numTopics+" topics: "+perplexity);
      }
      writer.close()

    }

  }  
}
